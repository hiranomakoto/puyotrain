"""時間のモデル(``TimedVersus``)の自己対戦の学習環境。

``selfplay.VersusVecEnv`` と同じく、相手の手番を環境の中で処理し、PPO からは1人用の並列環境に
見せる。時刻の進行・観測の計算・greedy の手は Rust の ``TimedBatch`` で全局分をまとめて並列に行う。
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .._puyo import TimedBatch
from ..spaces import action_space, observation_space
from .agents import Agent, GreedyAgent, PPOAgent
from .selfplay import OpponentPool, StackedActors, VersusVecEnv

_SEED_HIGH = 2**63


class _TimedCalls:
    """``TimedBatch`` の呼び出しを、環境の内訳としてプロファイラに記録する。

    プロファイラが付いていなければ(``env.timer`` が None)、そのまま呼ぶ。
    区分: observe(観測・特徴量の計算)、advance(手の決定と時刻の進行)、greedy(greedy の手)。
    """

    SECTIONS = {"observe": "env.observe", "act": "env.advance", "advance": "env.advance",
                "greedy": "env.greedy"}

    def __init__(self, env: "TimedVersusVecEnv", batch: TimedBatch):
        self._env = env
        self._batch = batch

    def __getattr__(self, name: str):
        fn = getattr(self._batch, name)
        section = self.SECTIONS.get(name)
        timer = self._env.timer
        if section is None or timer is None:
            return fn
        return timer.wrap(fn, section)


class TimedVersusVecEnv(VersusVecEnv):
    """時間のモデルの自己対戦環境。

    1ステップは「学習側が手を決めてから、次に操作待ちになるまで」。その間に相手が操作待ちに
    なれば、相手の手をまとめて求めて時刻を進める(相手は0手のことも、数手打つこともある)。

    報酬は ``VersusVecEnv`` と同じ(勝ち +1 / 負け −1、減っていく補助報酬)。補助報酬の
    「生んだおじゃま」は、このステップの間に学習側の連鎖が生んだ分。
    ``info["episode"]`` には ``result`` と ``opponent`` に加えて、学習側の最大連鎖 ``max_chain``
    と対局時間 ``seconds`` を入れる。

    相手のうち過去の自分(``pool`` の past)は、時間のモデル用の重み(観測の種類 timed)を前提に
    重みを束ねて1回で推論する。
    """

    def __init__(self, n_envs: int, pool: OpponentPool, safe_mask: bool, attack_coef: float,
                 potential_coef: float, gamma: float, anneal_steps: int, seed: int = 0,
                 start_step: int = 0, timing: Optional[dict] = None):
        self.render_mode = None  # VecEnv.__init__ が読む
        VecEnv.__init__(self, n_envs, observation_space(timed=True), action_space())
        self.batch = _TimedCalls(self, TimedBatch(n_envs, timing))
        self.pool = pool
        self.safe_mask = safe_mask
        self.attack_coef = attack_coef
        self.potential_coef = potential_coef
        self.gamma = gamma
        self.anneal_steps = anneal_steps
        self.steps = start_step
        self.rng = np.random.default_rng(seed)
        self.sides = [0] * n_envs
        self.opponents: list[tuple[str, Agent]] = [("", None)] * n_envs  # type: ignore[list-item]
        self.phi = [0.0] * n_envs
        self.last_attack = [0] * n_envs
        self.ep_return = [0.0] * n_envs
        self.ep_start = [0.0] * n_envs
        self.masks: list = [None] * n_envs
        self.actions: Optional[np.ndarray] = None
        self.timer = None  # プロファイリング用(profiling.SectionTimer)
        self._stacked_key: tuple = ()
        self._stacked: Optional[StackedActors] = None

    # ------------------------------------------------------------------ 対局の開始・観測

    def _new_game(self, i: int) -> None:
        seed = int(self.rng.integers(0, _SEED_HIGH))
        self.batch.reset(i, seed)
        self.sides[i] = int(self.rng.integers(2))
        self.opponents[i] = self.pool.sample()
        self.opponents[i][1].reset(seed % 2**32)
        self.phi[i] = 0.0
        self.last_attack[i] = 0
        self.ep_return[i] = 0.0
        self.ep_start[i] = time.time()

    def _observe(self, idx: list[int]) -> dict:
        obs = self.batch.observe(idx, [self.sides[i] for i in idx], "timed", self.safe_mask)
        for row, i in enumerate(idx):
            self.masks[i] = obs["action_mask"][row].astype(bool)
        return obs

    def reset(self):
        for i in range(self.num_envs):
            self._new_game(i)
        return self._observe(list(range(self.num_envs)))

    # ------------------------------------------------------------------ 相手の手

    def _opponent_turn(self, idx: list[int]) -> None:
        """相手が操作待ちの局 ``idx`` で、相手の手をまとめて求めて決めさせる。"""
        acts: dict[int, int] = {}
        greedy: list[int] = []
        batched: dict[int, list[int]] = defaultdict(list)
        for i in idx:
            agent = self.opponents[i][1]
            if isinstance(agent, PPOAgent):
                batched[id(agent)].append(i)
            elif isinstance(agent, GreedyAgent):
                greedy.append(i)
            else:
                acts[i] = agent.act(self.batch.game(i), 1 - self.sides[i])
        if greedy:
            seeds = [int(s) for s in self.rng.integers(0, 2**62, len(greedy))]
            players = [1 - self.sides[i] for i in greedy]
            for i, a in zip(greedy, self.batch.greedy(greedy, players, seeds)):
                acts[i] = a

        stacked = self._stacked_past()
        if stacked is not None:
            groups = {k: batched.pop(aid) for aid, k in stacked.index.items() if aid in batched}
            if groups:
                acts.update(self._stacked_actions(stacked, groups))

        for rows in batched.values():
            agent = self.opponents[rows[0]][1]
            obs = self.batch.observe(rows, [1 - self.sides[i] for i in rows], agent.obs_kind,
                                     agent.safe_mask)
            with self._timed("opp_infer"):
                chosen = agent.act_stacked(obs)
            acts.update(zip(rows, chosen))
        self.batch.act(idx, [1 - self.sides[i] for i in idx], [acts[i] for i in idx])

    def _stacked_actions(self, stacked: StackedActors,
                         groups: dict[int, list[int]]) -> dict[int, int]:
        """過去の自分(重みを束ねたもの)の手。groups: 方策の番号 → その方策が相手の局。"""
        any_agent = self.opponents[next(iter(groups.values()))[0]][1]
        obs = {k: self.batch.observe(rows, [1 - self.sides[i] for i in rows], any_agent.obs_kind,
                                     any_agent.safe_mask)
               for k, rows in groups.items()}
        n_models = len(stacked.index)
        width = max(len(rows) for rows in groups.values())
        fill = next(iter(obs.values()))
        arrays = {key: np.repeat(val[:1][None], n_models, axis=0).repeat(width, axis=1)
                  for key, val in fill.items()}  # 余りは適当な局面で埋める
        for k, o in obs.items():
            m = len(o["action_mask"])
            for key in arrays:
                arrays[key][k, :m] = o[key]
        with self._timed("opp_infer"):
            logits = stacked.logits(arrays)
            th = stacked._th
            masks = th.as_tensor(arrays["action_mask"].astype(bool), device=logits.device)
            choice = logits.masked_fill(~masks, -th.inf).argmax(-1).cpu().numpy()
        return {i: int(choice[k, b]) for k, rows in groups.items() for b, i in enumerate(rows)}

    # ------------------------------------------------------------------ 進行

    def step_wait(self):
        assert self.actions is not None
        n = self.num_envs
        everyone = list(range(n))
        self.batch.act(everyone, self.sides, [int(a) for a in self.actions])
        active = everyone
        while active:
            self.batch.advance(active)
            need = self.batch.needs_action()
            over = self.batch.is_over()
            waiting = [i for i in active if not over[i] and need[i, 1 - self.sides[i]]]
            if not waiting:
                break
            self._opponent_turn(waiting)
            active = waiting

        over = self.batch.is_over()
        winners = self.batch.winners()
        timed_out = self.batch.timed_out()
        now = self.batch.now()
        stats = self.batch.stats()
        live = [i for i in everyone if not over[i]]
        potential = dict(zip(live, self.batch.potential_chains(live,
                                                               [self.sides[i] for i in live])))
        alpha = self.anneal()
        rewards = np.zeros(n, dtype=np.float32)
        results = [0] * n
        for i in everyone:
            side = self.sides[i]
            attack = int(stats["attack_total"][i, side])
            r = alpha * self.attack_coef * (attack - self.last_attack[i])
            self.last_attack[i] = attack
            phi = 0.0 if over[i] else self.potential_coef * potential[i]
            r += alpha * (self.gamma * phi - self.phi[i])
            self.phi[i] = phi
            if over[i] and winners[i] in (0, 1):
                results[i] = 1 if winners[i] == side else -1
                r += results[i]
            rewards[i] = r
            self.ep_return[i] += r

        obs = self._observe(everyone)
        infos: list[dict[str, Any]] = [{} for _ in everyone]
        finished = [i for i in everyone if over[i]]
        for i in finished:
            side = self.sides[i]
            infos[i] = {
                "terminal_observation": {k: v[i].copy() for k, v in obs.items()},
                "TimeLimit.truncated": bool(timed_out[i]),
                "episode": {"r": self.ep_return[i], "l": int(stats["moves"][i, side]),
                            "t": round(time.time() - self.ep_start[i], 6),
                            "result": results[i], "opponent": self.opponents[i][0],
                            "max_chain": int(stats["max_chain"][i, side]),
                            "seconds": float(now[i]) / 60},
            }
            self._new_game(i)
        if finished:
            fresh = self._observe(finished)
            for row, i in enumerate(finished):
                for k in obs:
                    obs[k][i] = fresh[k][row]
        self.steps += n
        return obs, rewards, np.array(over, dtype=bool), infos
