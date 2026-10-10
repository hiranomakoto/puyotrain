"""方策ネットの特徴抽出器: 盤面を小さな CNN に通し、ネクスト等と結合する。"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class PuyoFeatures(BaseFeaturesExtractor):
    """観測 dict → 特徴ベクトル。

    - ``board[:, 0]``(自分の盤面、6平面 × 13段 × 6列)を 3x3 畳み込みに通す。
    - ``piece_planes`` が True なら、``pieces`` の各値(3組 × 軸・子 × 4色 = 24)を
      盤面全体に敷き詰めた平面として盤面に重ねてから畳み込む。各マスで「そこにある色」と
      「これから置く色」を直接比べられるようにするため。
    - ``pieces``(現在・ネクスト・ネクネクのワンホット)と ``ojama``(/30 で正規化)を結合する。
    - ``action_mask`` は特徴に使わない(MaskablePPO がマスクとして使う)。
    相手盤面は使わないので、1人用で学習した重みを対戦でもそのまま使える。
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_channels: tuple[int, ...] = (32, 64),
        features_dim: int = 256,
        piece_planes: bool = False,
    ):
        super().__init__(observation_space, features_dim)
        _, planes, height, width = observation_space["board"].shape
        n_pieces = int(np.prod(observation_space["pieces"].shape))
        self.piece_planes = piece_planes
        layers: list[nn.Module] = []
        c_in = planes + (n_pieces if piece_planes else 0)
        for c in cnn_channels:
            layers += [nn.Conv2d(c_in, c, kernel_size=3, padding=1), nn.ReLU()]
            c_in = c
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        n_cnn = c_in * height * width
        n_extra = n_pieces + observation_space["ojama"].shape[0]
        self.head = nn.Sequential(nn.Linear(n_cnn + n_extra, features_dim), nn.ReLU())

    def forward(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        board = obs["board"][:, 0].float()
        pieces = obs["pieces"].flatten(1).float()
        if self.piece_planes:
            _, _, height, width = board.shape
            board = th.cat([board, pieces[:, :, None, None].expand(-1, -1, height, width)], dim=1)
        x = self.cnn(board)
        extra = th.cat([pieces, obs["ojama"].float() / 30.0], dim=1)
        return self.head(th.cat([x, extra], dim=1))


class PuyoActionFeatures(BaseFeaturesExtractor):
    """手ごとに点数をつける特徴抽出器(観測に ``action_features`` が必要)。

    - 盤面は ``PuyoFeatures`` で文脈ベクトル(``features_dim``)にする。
    - 22通りの手それぞれについて、その手の結果の特徴量(``Game.action_features``)と
      文脈を縮めたベクトルを、全手で共通の小さな MLP(``action_hidden``)に通して点数にする。
    - 出力は ``[22手の点数, 文脈ベクトル]``。``PuyoActionPolicy`` は先頭22次元をそのまま
      ロジットに使い、価値は全体から計算する。
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_channels: tuple[int, ...] = (32, 64),
        features_dim: int = 256,
        piece_planes: bool = True,
        action_hidden: tuple[int, ...] = (64, 64),
        context_dim: int = 32,
    ):
        n_actions, n_features = observation_space["action_features"].shape
        super().__init__(observation_space, n_actions + features_dim)
        self.n_actions = n_actions
        self.board = PuyoFeatures(observation_space, cnn_channels, features_dim, piece_planes)
        self.context = nn.Sequential(nn.Linear(features_dim, context_dim), nn.ReLU())
        layers: list[nn.Module] = []
        c_in = n_features + context_dim
        for h in action_hidden:
            layers += [nn.Linear(c_in, h), nn.ReLU()]
            c_in = h
        layers.append(nn.Linear(c_in, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        ctx = self.board(obs)
        feats = obs["action_features"].float()
        c = self.context(ctx)[:, None, :].expand(-1, feats.shape[1], -1)
        scores = self.scorer(th.cat([feats, c], dim=-1)).squeeze(-1)
        return th.cat([scores, ctx], dim=1)


class TakeFirst(nn.Module):
    """先頭 ``n`` 次元だけを取り出す(パラメータなし)。"""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x[:, : self.n]


class PuyoActionPolicy(MaskableMultiInputActorCriticPolicy):
    """``PuyoActionFeatures`` がつけた手ごとの点数をそのままロジットにする方策。

    ``net_arch`` の ``pi`` は空にする(特徴抽出器の出力がそのまま方策側に渡るように)。
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = TakeFirst(self.action_space.n)
        # 最初はほぼ一様な方策から始める(SB3 が action_net に使う 0.01 と同じ)
        last = self.features_extractor.scorer[-1]
        nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
