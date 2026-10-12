"""エージェント。1人用 (``Game``) と対戦 (``Versus`` の片側) の両方で使える。

``act(state, player)`` の ``state`` は ``Game`` か ``Versus``。``Game`` のとき ``player`` は無視する。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .._puyo import (
    NUM_LOOK_ACTION_FEATURES,
    NUM_TIMED_ACTION_FEATURES,
    Game,
    TimedVersus,
    Versus,
)
from ..envs import solo_observation, versus_observation

State = Union[Game, Versus, TimedVersus]


def _mask(state: State, player: int) -> int:
    return state.legal_actions() if isinstance(state, Game) else state.legal_actions(player)


class Agent:
    """エージェントの共通インターフェース。"""

    kind = "agent"

    def reset(self, seed: int) -> None:
        """対局の開始時に呼ぶ(乱数を対局ごとに固定するため)。"""

    def act(self, state: State, player: int = 0) -> int:
        raise NotImplementedError

    def describe(self) -> dict:
        """リプレイのメタ情報に残す内容。"""
        return {"type": self.kind}


class RandomAgent(Agent):
    """合法手から一様に選ぶ。"""

    kind = "random"

    def __init__(self, seed: int = 0):
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng([self.base_seed, seed])

    def act(self, state: State, player: int = 0) -> int:
        m = _mask(state, player)
        legal = [i for i in range(22) if m >> i & 1]
        return int(legal[self.rng.integers(len(legal))])


class GreedyAgent(Agent):
    """2手読みで合計得点が最大の手(同点処理は puyo-core/src/bot.rs)。"""

    kind = "greedy"

    def __init__(self, seed: int = 0):
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng([self.base_seed, seed])

    def act(self, state: State, player: int = 0) -> int:
        s = int(self.rng.integers(0, 2**63))
        a = state.greedy_action(s) if isinstance(state, Game) else state.greedy_action(player, s)
        if a is None:
            raise RuntimeError("no legal action")
        return a


def _trained_with_safe_mask(path: Path) -> bool:
    """チェックポイントの記録(.json)から、学習時に自滅禁止マスクを使っていたかを読む。"""
    meta = path.with_suffix(".json")
    if not meta.exists():
        return False
    cfg = json.loads(meta.read_text(encoding="utf-8")).get("config", {})
    return bool(cfg.get("env", {}).get("safe_mask", False))


_STEP_RE = re.compile(r"step_(\d+)")


def checkpoint_step(path: Union[str, Path]) -> Optional[int]:
    """チェックポイントのファイル名からステップ数を取り出す。"""
    m = _STEP_RE.search(Path(path).stem)
    return int(m.group(1)) if m else None


class PPOAgent(Agent):
    """学習済み MaskablePPO。行動マスク付きで推論する。

    ``PPOAgent(path)`` でチェックポイントを読み込む。学習中のモデルを直接使うときは
    ``PPOAgent(path, model=model)``(ファイルは読まない)。

    ``safe_mask`` が True なら、自滅する手(``safe_actions``)を除いて選ぶ。None なら、
    チェックポイント横の .json に記録された学習時の設定 ``env.safe_mask`` に合わせる。

    モデルの観測に ``action_features`` があれば(``policy.action_features`` で学習した重み)、
    観測に手ごとの結果の特徴量を加えて打つ。

    モデルの観測に ``versus_info`` があれば(対戦用に学習した重み)、対戦用の特徴量で打つ。
    この重みは対戦専用(1人用の ``Game`` では打てない)。それ以外の重みは対戦でも1人用と
    同じ入力(予告おじゃまを考えない特徴量)で打つ。

    時間のモデル(``TimedVersus``)では、重みの種類 ``obs_kind``(solo / versus / timed)に
    合わせた観測を ``TimedVersus.observe`` で作る。時間のモデル用の重み(timed)は
    ``TimedVersus`` 専用。
    """

    kind = "ppo"

    def __init__(
        self,
        path: Union[str, Path],
        device: str = "cpu",
        deterministic: bool = True,
        model=None,
        safe_mask: Optional[bool] = None,
    ):
        self.path = Path(path)
        if model is None:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(str(self.path), device=device)
        self.model = model
        self.deterministic = deterministic
        self.step = checkpoint_step(self.path)
        self.safe_mask = _trained_with_safe_mask(self.path) if safe_mask is None else safe_mask
        spaces = model.observation_space.spaces
        self.action_features = "action_features" in spaces
        self.versus_features = "versus_info" in spaces
        n_features = spaces["action_features"].shape[1] if self.action_features else 0
        self.timed_features = n_features in (NUM_TIMED_ACTION_FEATURES, NUM_LOOK_ACTION_FEATURES)
        if n_features == NUM_LOOK_ACTION_FEATURES:
            self.obs_kind: Optional[str] = "look"
        elif self.timed_features:
            self.obs_kind = "timed"
        elif self.versus_features:
            self.obs_kind = "versus"
        elif self.action_features:
            self.obs_kind = "solo"
        else:
            self.obs_kind = None

    def observe(self, state: State, player: int = 0) -> dict:
        """このモデルが学習したときと同じ形の観測を作る。"""
        if isinstance(state, TimedVersus):
            if self.obs_kind is None:
                return state.observation(player)
            return state.observe(player, self.obs_kind, self.safe_mask)
        if self.timed_features:
            raise ValueError("this checkpoint was trained for the time-based rules (TimedVersus)")
        if isinstance(state, Game):
            if self.versus_features:
                raise ValueError("this checkpoint was trained for versus games only")
            return solo_observation(state, self.safe_mask, self.action_features)
        return versus_observation(state, player, self.safe_mask, self.action_features,
                                  self.versus_features)

    def act(self, state: State, player: int = 0) -> int:
        obs = self.observe(state, player)
        mask = obs["action_mask"].astype(bool)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return int(action)

    def act_batch(self, observations: list[dict]) -> list[int]:
        """複数局面をまとめて推論する(評価・自己対戦の相手用)。

        決定的なとき(既定)は SB3 の ``predict`` を通さず、方策ネットに直接流して
        マスク付きの argmax を取る(``predict`` と同じ手。呼び出しごとの前処理・分布の
        組み立てが無い分速い)。確率的なときは ``predict`` を使う。
        """
        obs = {k: np.stack([o[k] for o in observations]) for k in observations[0]}
        return self.act_stacked(obs)

    def act_stacked(self, obs: dict) -> list[int]:
        """``act_batch`` と同じ。観測は件数の次元を先頭に持つ配列の dict(``TimedBatch.observe``)。"""
        masks = obs["action_mask"].astype(bool)
        if self.deterministic:
            return self._argmax_actions(obs, masks)
        actions, _ = self.model.predict(obs, action_masks=masks, deterministic=False)
        return [int(a) for a in actions]

    def _argmax_actions(self, obs: dict, masks: np.ndarray) -> list[int]:
        import torch as th

        policy = self.model.policy
        with th.no_grad():
            t = {k: th.as_tensor(v, device=policy.device) for k, v in obs.items()}
            features = policy.extract_features(t)
            if isinstance(features, tuple):  # 方策と価値で特徴抽出を共有していない場合
                features = features[0]
            logits = policy.action_net(policy.mlp_extractor.forward_actor(features))
            logits = logits.masked_fill(~th.as_tensor(masks, device=policy.device), -th.inf)
            return logits.argmax(dim=1).tolist()

    def describe(self) -> dict:
        return {"type": self.kind, "checkpoint": str(self.path), "step": self.step}


def make_agent(spec: str, seed: int = 0, device: str = "cpu") -> Agent:
    """``random`` / ``greedy`` / チェックポイント(.zip)のパスからエージェントを作る。"""
    if spec == "random":
        return RandomAgent(seed)
    if spec == "greedy":
        return GreedyAgent(seed)
    p = Path(spec)
    if not p.exists():
        raise FileNotFoundError(f"agent must be 'random', 'greedy' or a checkpoint path: {spec}")
    return PPOAgent(p, device=device)


def agent_label(agent: Agent) -> str:
    """ファイル名・動画用の短い名前。"""
    if isinstance(agent, PPOAgent):
        return f"step_{agent.step:010d}" if agent.step is not None else agent.path.stem
    return f"baseline_{agent.kind}"
