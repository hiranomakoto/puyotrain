"""Gymnasium 互換の環境。"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from ._puyo import Game, Versus
from .spaces import action_space, observation_space

_SEED_HIGH = 2**63


def solo_observation(game: Game, safe_mask: bool = False, action_features: bool = False) -> dict:
    """1人用の観測。``safe_mask`` なら行動マスクを自滅しない手に絞り、
    ``action_features`` なら手ごとの結果の特徴量を加える(``PuyoSoloEnv`` と同じ形)。"""
    obs = game.observation()
    if action_features:
        feats = game.action_features()
        obs["action_features"] = feats
        if safe_mask:
            # 特徴量の legal・dead・survivable から作る(Game.safe_action_mask と同じ結果)
            safe = (feats[:, 0] > 0) & (feats[:, 1] == 0) & (feats[:, 10] > 0)
            if safe.any():
                obs["action_mask"] = safe.astype(np.int8)
    elif safe_mask:
        obs["action_mask"] = game.safe_action_mask()
    return obs


class PuyoSoloEnv(gym.Env):
    """1人用。報酬はその手で得た得点 × reward_scale。窒息で terminated。

    Args:
        max_steps: この手数で truncated にする。None なら無制限。
        reward_scale: 報酬に掛ける係数(学習では 0.001 など)。
        death_penalty: 窒息した手の報酬から引く値(reward_scale は掛けない)。
        survival_bonus: 窒息しなかった手ごとに足す値(reward_scale は掛けない)。
        potential_coef: 潜在連鎖による報酬整形の係数。Φ(s) = potential_coef × 潜在連鎖数
            として、毎手 ``potential_gamma × Φ(置いた後) − Φ(置く前)`` を報酬に足す
            (窒息したら Φ(置いた後) = 0)。ポテンシャルに基づく整形なので最適方策は変わらない。
        potential_gamma: 整形に使う割引率。学習の gamma と同じ値にする。
        safe_mask: True なら自滅する手(``Game.safe_actions``)を行動マスクで禁止する。
        action_features: True なら観測に手ごとの結果の特徴量(``Game.action_features``)を加える。
        render_mode: None または "ansi"。

    ``reset(options={"game_seed": s})`` でゲームのシードを直接指定できる
    (固定シード評価用)。info には ``max_chain``(このエピソードの最大連鎖数)が入る。
    ``potential_coef`` が0でなければ ``potential_chain``(置いた後の潜在連鎖数)も入る。
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        max_steps: Optional[int] = None,
        reward_scale: float = 1.0,
        death_penalty: float = 0.0,
        survival_bonus: float = 0.0,
        potential_coef: float = 0.0,
        potential_gamma: float = 0.99,
        safe_mask: bool = False,
        action_features: bool = False,
        render_mode: Optional[str] = None,
    ):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.death_penalty = death_penalty
        self.survival_bonus = survival_bonus
        self.potential_coef = potential_coef
        self.potential_gamma = potential_gamma
        self.safe_mask = safe_mask
        self.action_features = action_features
        self.observation_space = observation_space(action_features)
        self.action_space = action_space()
        self.game: Optional[Game] = None
        self.max_chain = 0
        self.potential = 0
        self._mask: Optional[np.ndarray] = None

    def _observation(self) -> dict:
        obs = solo_observation(self.game, self.safe_mask, self.action_features)
        self._mask = obs["action_mask"].astype(bool)  # action_masks() で再計算しないよう保持
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options and options.get("game_seed") is not None:
            game_seed = int(options["game_seed"])
        else:
            game_seed = int(self.np_random.integers(0, _SEED_HIGH))
        self.game = Game(game_seed)
        self.max_chain = 0
        self.potential = 0
        return self._observation(), {"game_seed": game_seed}

    def step(self, action):
        if self.game is None:
            raise RuntimeError("call reset() before step()")
        score, chains, all_clear, dead = self.game.step(int(action))
        self.max_chain = max(self.max_chain, chains)
        terminated = dead
        truncated = (
            not terminated and self.max_steps is not None and self.game.turn >= self.max_steps
        )
        info = {
            "score": score,
            "chains": chains,
            "all_clear": all_clear,
            "total_score": self.game.score,
            "max_chain": self.max_chain,
        }
        reward = float(score) * self.reward_scale
        reward += -self.death_penalty if dead else self.survival_bonus
        if self.potential_coef:
            potential = 0 if dead else self.game.potential_chain()[0]
            reward += self.potential_coef * (self.potential_gamma * potential - self.potential)
            self.potential = potential
            info["potential_chain"] = potential
        return self._observation(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """行動マスク(sb3-contrib の MaskablePPO 互換)。``safe_mask`` なら自滅する手も除く。"""
        if self.game is None:
            raise RuntimeError("call reset() before action_masks()")
        return self._mask

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
