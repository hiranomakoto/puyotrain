"""自己対戦の学習環境: 相手の手番を環境の中で処理し、PPO からは1人用の並列環境に見せる。

- 対局ごとに、相手を ``OpponentPool`` から選ぶ(greedy、1人プレイの完成版、過去の自分)。
- 学習側の席(0 か 1)も対局ごとにランダムに選ぶ。
- 相手が PPO の重みのときは、同じ相手と打っている環境の分をまとめて1回で推論する。
  過去の自分(形が同じ方策)は、重みを束ねて全員分を1回で推論する(``StackedActors``)。
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
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
    """対戦相手のプール。カテゴリを重みで選び、その中から1人選ぶ。

    カテゴリは greedy、固定の相手(``fixed`` の名前ごとに1つ)、past(過去の自分)。
    空のカテゴリと、重みが無いカテゴリは選ばない(学習の最初は past が空)。

    ``fixed`` は {名前: チェックポイントのパス}。以前の形(パス1つ)を渡すと名前は solo になる。
    """

    def __init__(self, weights: dict[str, float], fixed: dict[str, str] | str | None = None,
                 device: str = "cpu", max_snapshots: int = 10, seed: int = 0):
        self.weights = dict(weights)
        self.device = device
        self.max_snapshots = max_snapshots
        self.rng = np.random.default_rng(seed)
        if isinstance(fixed, (str, Path)):
            fixed = {"solo": str(fixed)}
        self.members: dict[str, list[Agent]] = {"greedy": [GreedyAgent(seed)]}
        for name, path in (fixed or {}).items():
            self.members[name] = [PPOAgent(path, device=device)] if path else []
        self.members["past"] = []

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


def _shape_key(agent: PPOAgent) -> tuple:
    """束ねられるかの判定用: 観測の種類と、方策のパラメータの名前と形。"""
    return (agent.obs_kind,
            tuple((n, tuple(p.shape)) for n, p in agent.model.policy.named_parameters()))


class StackedActors:
    """形が同じ複数の方策(過去の自分)の重みを束ね、``torch.func.vmap`` で1回の推論にまとめる。

    ``logits(obs)`` の ``obs`` は各キーが ``(K, B, ...)``(K = 方策の数、B = 1方策あたりの局面数)。
    方策ごとに局面数が違うときは、呼び出し側で B にそろえて(余りは適当な局面で埋めて)渡す。
    """

    def __init__(self, agents: list[PPOAgent]):
        import copy

        import torch as th
        from torch import nn
        from torch.func import functional_call, stack_module_state, vmap

        class Actor(nn.Module):
            """方策ネットの行動ロジットまで(価値は計算しない)。"""

            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            def forward(self, obs):
                f = self.policy.extract_features(obs)
                if isinstance(f, tuple):
                    f = f[0]
                return self.policy.action_net(self.policy.mlp_extractor.forward_actor(f))

        actors = [Actor(a.model.policy) for a in agents]
        self.device = agents[0].model.policy.device
        self.params, self.buffers = stack_module_state(actors)
        base = copy.deepcopy(actors[0]).to("meta")

        def call(params, buffers, obs):
            return functional_call(base, (params, buffers), (obs,))

        self._fn = vmap(call)
        self._th = th
        self.index = {id(a): k for k, a in enumerate(agents)}

    def logits(self, obs: dict[str, np.ndarray]):
        th = self._th
        with th.no_grad():
            t = {k: th.as_tensor(v, device=self.device) for k, v in obs.items()}
            return self._fn(self.params, self.buffers, t)


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
        self.timer = None  # プロファイリング用(profiling.SectionTimer)。相手の推論時間を測る
        self._stacked_key: tuple = ()
        self._stacked: list[StackedActors] = []

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

    def _stacked_past(self) -> list[StackedActors]:
        """過去の自分を、形が同じ方策(観測の種類とパラメータの形が同じもの)ごとに束ねたもの。
        2人以上いるグループだけを束ねる(1人のグループは個別に推論する)。プールが変わったら作り直す。

        初期の過去の自分(``pool.initial_past``)と、学習中に加わる今の重みとで入力の形が違う場合
        (例: 先読みなしの timed と先読みつきの look)があるので、形ごとに分ける。
        """
        past = self.pool.members["past"]
        key = tuple(id(a) for a in past)
        if key != self._stacked_key:
            self._stacked_key = key
            groups: dict[tuple, list[PPOAgent]] = defaultdict(list)
            for a in past:
                groups[_shape_key(a)].append(a)
            self._stacked = [StackedActors(g) for g in groups.values() if len(g) >= 2]
        return self._stacked

    @contextmanager
    def _timed(self, name: str):
        if self.timer is None:
            yield
        else:
            with self.timer.section(name, sync=True):
                yield

    def _opponent_actions(self) -> list[int]:
        acts: list[int] = [0] * self.num_envs
        batched: dict[int, list[int]] = defaultdict(list)
        for i, (_, agent) in enumerate(self.opponents):
            if isinstance(agent, PPOAgent):
                batched[id(agent)].append(i)
            else:
                acts[i] = agent.act(self.games[i], 1 - self.sides[i])
        for stacked in self._stacked_past():
            # 過去の自分は、形が同じグループごとに全員分を1回の推論で(方策ごとの局面数を B にそろえて束ねる)
            groups = {k: batched.pop(aid) for aid, k in stacked.index.items() if aid in batched}
            if groups:
                obs = {k: [self.opponents[idxs[0]][1].observe(self.games[i], 1 - self.sides[i])
                           for i in idxs] for k, idxs in groups.items()}
                n_models = len(stacked.index)
                width = max(len(v) for v in obs.values())
                fill = next(iter(obs.values()))[0]
                arrays = {key: np.empty((n_models, width, *np.shape(val)),
                                        dtype=np.asarray(val).dtype)
                          for key, val in fill.items()}
                for k in range(n_models):
                    rows = obs.get(k, [])
                    for b in range(width):
                        o = rows[b] if b < len(rows) else fill
                        for key in arrays:
                            arrays[key][k, b] = o[key]
                with self._timed("opp_infer"):
                    logits = stacked.logits(arrays)
                    masks = stacked._th.as_tensor(arrays["action_mask"].astype(bool),
                                                  device=logits.device)
                    choice = logits.masked_fill(~masks, -stacked._th.inf).argmax(-1).cpu().numpy()
                for k, idxs in groups.items():
                    for b, i in enumerate(idxs):
                        acts[i] = int(choice[k, b])
        for idxs in batched.values():
            agent = self.opponents[idxs[0]][1]
            obs_list = [agent.observe(self.games[i], 1 - self.sides[i]) for i in idxs]
            with self._timed("opp_infer"):
                batch_acts = agent.act_batch(obs_list)
            for i, a in zip(idxs, batch_acts):
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
