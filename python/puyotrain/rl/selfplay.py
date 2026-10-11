"""自己対戦の学習環境: 相手の手番を環境の中で処理し、PPO からは1人用の並列環境に見せる。

- 対局ごとに、相手を ``OpponentPool`` から選ぶ(greedy、1人プレイの完成版、過去の自分)。
- 学習側の席(0 か 1)も対局ごとにランダムに選ぶ。
- 相手が PPO の重みのときは、同じ相手と打っている環境の分をまとめて1回で推論する。
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .._puyo import Versus
from ..envs import versus_observation
from ..spaces import action_space, observation_space
from .agents import Agent, GreedyAgent, PPOAgent

_SEED_HIGH = 2**63


class OpponentPool:
    """対戦相手のプール。カテゴリ(greedy / solo / past)を重みで選び、その中から1人選ぶ。

    空のカテゴリは選ばない(学習の最初は past が空)。
    """

    def __init__(self, weights: dict[str, float], solo_checkpoint: Optional[str],
                 device: str = "cpu", max_snapshots: int = 10, seed: int = 0):
        self.weights = dict(weights)
        self.device = device
        self.max_snapshots = max_snapshots
        self.rng = np.random.default_rng(seed)
        self.members: dict[str, list[Agent]] = {"greedy": [GreedyAgent(seed)], "solo": [],
                                                "past": []}
        if solo_checkpoint:
            self.members["solo"].append(PPOAgent(solo_checkpoint, device=device))

    def add_snapshot(self, path: Path) -> None:
        """過去の自分を追加する。上限を超えたら古いものから外す。"""
        self.members["past"].append(PPOAgent(path, device=self.device))
        del self.members["past"][:-self.max_snapshots]

    def snapshot_paths(self) -> list[str]:
        return [str(a.path) for a in self.members["past"]]

    def sample(self) -> tuple[str, Agent]:
        cats = [c for c, m in self.members.items() if m and self.weights.get(c, 0) > 0]
        w = np.array([self.weights[c] for c in cats], dtype=float)
        cat = cats[self.rng.choice(len(cats), p=w / w.sum())]
        group = self.members[cat]
        return cat, group[self.rng.integers(len(group))]


class VersusVecEnv(VecEnv):
    """自己対戦の並列環境(``n_envs`` 局を同時に進める)。

    報酬(学習側から見て):

    - 勝ち +1、負け −1、引き分け・打ち切りは 0。
    - 補助報酬(係数は学習の進み具合で直線的に減らし、``anneal_steps`` で0にする):
      - この手の連鎖で生んだおじゃま(相殺分も含む) × ``attack_coef``
      - 潜在連鎖の整形 ``gamma × Φ(後) − Φ(前)``、Φ = ``potential_coef`` × 潜在連鎖数(終局で0)

    ``info["episode"]`` には SB3 の統計(r, l, t)に加えて、勝敗 ``result``(1 / 0 / −1)と
    相手のカテゴリ ``opponent`` を入れる。
    """

    def __init__(self, n_envs: int, pool: OpponentPool, max_turns: int, safe_mask: bool,
                 attack_coef: float, potential_coef: float, gamma: float, anneal_steps: int,
                 seed: int = 0, start_step: int = 0):
        self.render_mode = None  # VecEnv.__init__ が読む
        super().__init__(n_envs, observation_space(versus=True), action_space())
        self.pool = pool
        self.max_turns = max_turns
        self.safe_mask = safe_mask
        self.attack_coef = attack_coef
        self.potential_coef = potential_coef
        self.gamma = gamma
        self.anneal_steps = anneal_steps
        self.steps = start_step
        self.rng = np.random.default_rng(seed)
        self.games: list[Versus] = [None] * n_envs  # type: ignore[list-item]
        self.sides = [0] * n_envs
        self.opponents: list[tuple[str, Agent]] = [("", None)] * n_envs  # type: ignore[list-item]
        self.phi = [0.0] * n_envs
        self.ep_return = [0.0] * n_envs
        self.ep_start = [0.0] * n_envs
        self.masks = [None] * n_envs
        self.actions: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ 進行

    def anneal(self) -> float:
        """補助報酬の係数に掛ける値(1 → 0)。"""
        if self.anneal_steps <= 0:
            return 0.0
        return max(0.0, 1.0 - self.steps / self.anneal_steps)

    def _observe(self, i: int) -> dict:
        obs = versus_observation(self.games[i], self.sides[i], self.safe_mask,
                                 versus_features=True)
        self.masks[i] = obs["action_mask"].astype(bool)
        return obs

    def _new_game(self, i: int) -> dict:
        seed = int(self.rng.integers(0, _SEED_HIGH))
        self.games[i] = Versus(seed)
        self.sides[i] = int(self.rng.integers(2))
        self.opponents[i] = self.pool.sample()
        self.opponents[i][1].reset(seed % 2**32)
        self.phi[i] = 0.0
        self.ep_return[i] = 0.0
        self.ep_start[i] = time.time()
        return self._observe(i)

    def _stack(self, observations: list[dict]) -> dict:
        return {k: np.stack([o[k] for o in observations]) for k in observations[0]}

    def _opponent_actions(self) -> list[int]:
        acts: list[int] = [0] * self.num_envs
        batched: dict[int, list[int]] = defaultdict(list)
        for i, (_, agent) in enumerate(self.opponents):
            if isinstance(agent, PPOAgent):
                batched[id(agent)].append(i)
            else:
                acts[i] = agent.act(self.games[i], 1 - self.sides[i])
        for idxs in batched.values():
            agent = self.opponents[idxs[0]][1]
            obs = [agent.observe(self.games[i], 1 - self.sides[i]) for i in idxs]
            for i, a in zip(idxs, agent.act_batch(obs)):
                acts[i] = a
        return acts

    def reset(self):
        return self._stack([self._new_game(i) for i in range(self.num_envs)])

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self):
        assert self.actions is not None
        opp = self._opponent_actions()
        alpha = self.anneal()
        observations, rewards, dones, infos = [], [], [], []
        for i, v in enumerate(self.games):
            side = self.sides[i]
            pair = [0, 0]
            pair[side] = int(self.actions[i])
            pair[1 - side] = opp[i]
            res = v.step(*pair)[side]
            terminated = v.is_over
            truncated = not terminated and v.turn >= self.max_turns
            r = alpha * self.attack_coef * res["attack"]
            phi = 0.0 if terminated else self.potential_coef * v.potential_chain(side)[0]
            r += alpha * (self.gamma * phi - self.phi[i])
            self.phi[i] = phi
            result = 0
            if terminated and not v.is_draw:
                result = 1 if v.winner == side else -1
                r += result
            self.ep_return[i] += r
            info: dict[str, Any] = {}
            obs = self._observe(i)
            if terminated or truncated:
                info["terminal_observation"] = obs
                info["TimeLimit.truncated"] = truncated
                info["episode"] = {"r": self.ep_return[i], "l": v.turn,
                                   "t": round(time.time() - self.ep_start[i], 6),
                                   "result": result, "opponent": self.opponents[i][0]}
                obs = self._new_game(i)
            observations.append(obs)
            rewards.append(r)
            dones.append(terminated or truncated)
            infos.append(info)
        self.steps += self.num_envs
        return (self._stack(observations), np.array(rewards, dtype=np.float32),
                np.array(dones), infos)

    def action_masks(self) -> list[np.ndarray]:
        return list(self.masks)

    # ------------------------------------------------------------------ VecEnv の残り

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [getattr(self, attr_name)] * len(list(self._get_indices(indices)))

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def has_attr(self, attr_name: str) -> bool:
        return hasattr(self, attr_name)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            masks = self.action_masks()
            return [masks[i] for i in self._get_indices(indices)]
        result = getattr(self, method_name)(*method_args, **method_kwargs)
        return [result] * len(list(self._get_indices(indices)))

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * len(list(self._get_indices(indices)))

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs

    def get_images(self):
        return [None] * self.num_envs
