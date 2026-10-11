"""エージェント。1人用 (``Game``) と対戦 (``Versus`` の片側) の両方で使える。

``act(state, player)`` の ``state`` は ``Game`` か ``Versus``。``Game`` のとき ``player`` は無視する。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .._puyo import Game, Versus
from ..envs import solo_observation, versus_observation

State = Union[Game, Versus]


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

    対戦では、自滅判定と特徴量は予告おじゃまの降下を考えない(1人用で学習した重みと同じ入力)。
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
        self.action_features = "action_features" in model.observation_space.spaces

    def act(self, state: State, player: int = 0) -> int:
        if isinstance(state, Game):
            obs = solo_observation(state, self.safe_mask, self.action_features)
        else:
            obs = versus_observation(state, player, self.safe_mask, self.action_features)
        mask = obs["action_mask"].astype(bool)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=self.deterministic)
        return int(action)

    def act_batch(self, observations: list[dict]) -> list[int]:
        """複数局面をまとめて推論する(評価の高速化用)。"""
        obs = {k: np.stack([o[k] for o in observations]) for k in observations[0]}
        masks = obs["action_mask"].astype(bool)
        actions, _ = self.model.predict(obs, action_masks=masks, deterministic=self.deterministic)
        return [int(a) for a in actions]

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
