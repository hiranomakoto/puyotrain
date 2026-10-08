"""Gymnasium 互換の環境。"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from ._puyo import Game, Versus
from .spaces import action_space, observation_space

_SEED_HIGH = 2**63


class PuyoSoloEnv(gym.Env):
    """1人用。報酬はその手で得た得点。窒息で terminated。

    Args:
        max_steps: この手数で truncated にする。None なら無制限。
        render_mode: None または "ansi"。
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_steps: Optional[int] = None, render_mode: Optional[str] = None):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.observation_space = observation_space()
        self.action_space = action_space()
        self.game: Optional[Game] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(0, _SEED_HIGH))
        self.game = Game(game_seed)
        return self.game.observation(), {"game_seed": game_seed}

    def step(self, action):
        if self.game is None:
            raise RuntimeError("call reset() before step()")
        score, chains, all_clear, dead = self.game.step(int(action))
        terminated = dead
        truncated = (
            not terminated and self.max_steps is not None and self.game.turn >= self.max_steps
        )
        info = {"chains": chains, "all_clear": all_clear, "total_score": self.game.score}
        return self.game.observation(), float(score), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """合法手マスク(sb3-contrib の MaskablePPO 互換)。"""
        if self.game is None:
            raise RuntimeError("call reset() before action_masks()")
        return self.game.action_mask().astype(bool)

    def render(self):
        if self.render_mode == "ansi" and self.game is not None:
            return self.game.field_str()
        return None


class PuyoVersusEnv:
    """2人対戦(同期ターン制)。Gymnasium の Env と同じ形の API を持つ。

    ``step(a1, a2)`` は ``((obs1, obs2), (r1, r2), terminated, truncated, info)`` を返す。
    報酬は勝ち +1、負け -1、引き分け・打ち切り・途中は 0。観測は各プレイヤー視点(自分が先)。

    Args:
        max_turns: このターン数で truncated にする。None なら無制限。
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_turns: Optional[int] = None, render_mode: Optional[str] = None):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.max_turns = max_turns
        self.observation_space = observation_space()
        self.action_space = action_space()
        self.np_random = np.random.default_rng()
        self.game: Optional[Versus] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        game_seed = int(self.np_random.integers(0, _SEED_HIGH))
        self.game = Versus(game_seed)
        return self._observations(), {"game_seed": game_seed}

    def _observations(self):
        assert self.game is not None
        return self.game.observation(0), self.game.observation(1)

    def step(self, action1, action2):
        if self.game is None:
            raise RuntimeError("call reset() before step()")
        results = self.game.step(int(action1), int(action2))
        terminated = self.game.is_over
        truncated = (
            not terminated and self.max_turns is not None and self.game.turn >= self.max_turns
        )
        rewards = (0.0, 0.0)
        winner = self.game.winner
        if winner == 0:
            rewards = (1.0, -1.0)
        elif winner == 1:
            rewards = (-1.0, 1.0)
        info: dict[str, Any] = {
            "players": results,
            "winner": winner,
            "draw": self.game.is_draw,
            "turn": self.game.turn,
        }
        return self._observations(), rewards, terminated, truncated, info

    def action_masks(self) -> tuple[np.ndarray, np.ndarray]:
        if self.game is None:
            raise RuntimeError("call reset() before action_masks()")
        return self.game.action_mask(0).astype(bool), self.game.action_mask(1).astype(bool)

    def render(self):
        if self.render_mode != "ansi" or self.game is None:
            return None
        left = self.game.field_str(0).splitlines()
        right = self.game.field_str(1).splitlines()
        p0, p1 = self.game.pending_ojama()
        lines = [f"P1 ojama={p0:<3}  P2 ojama={p1}"]
        lines += [f"{a}    {b}" for a, b in zip(left, right)]
        return "\n".join(lines) + "\n"

    def close(self):
        pass
