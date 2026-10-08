"""Python 側の1秒あたりの手数を計測する。 `python python/benchmarks/bench_env.py`"""

import time

import numpy as np

from puyotrain import Game, PuyoSoloEnv, PuyoVersusEnv

BUDGET = 2.0


def report(name, n, elapsed):
    print(f"{name:<42} {n / elapsed:>12,.0f} /s")


def bench_raw_game():
    rng = np.random.default_rng(0)
    seed, n = 0, 0
    g = Game(seed)
    t = time.perf_counter()
    while time.perf_counter() - t < BUDGET:
        for _ in range(1000):
            if g.is_over:
                seed += 1
                g = Game(seed)
            mask = g.legal_actions()
            legal = [i for i in range(22) if mask >> i & 1]
            g.step(legal[rng.integers(len(legal))])
            n += 1
    report("Game.step (no observation)", n, time.perf_counter() - t)


def bench_solo_env():
    env = PuyoSoloEnv()
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    n = 0
    t = time.perf_counter()
    while time.perf_counter() - t < BUDGET:
        for _ in range(1000):
            legal = np.flatnonzero(obs["action_mask"])
            obs, _, term, trunc, _ = env.step(legal[rng.integers(len(legal))])
            n += 1
            if term or trunc:
                obs, _ = env.reset()
    report("PuyoSoloEnv.step (with observation)", n, time.perf_counter() - t)


def bench_versus_env():
    env = PuyoVersusEnv()
    rng = np.random.default_rng(0)
    (o0, o1), _ = env.reset(seed=0)
    n = 0
    t = time.perf_counter()
    while time.perf_counter() - t < BUDGET:
        for _ in range(1000):
            l0 = np.flatnonzero(o0["action_mask"])
            l1 = np.flatnonzero(o1["action_mask"])
            (o0, o1), _, term, trunc, _ = env.step(l0[rng.integers(len(l0))], l1[rng.integers(len(l1))])
            n += 1
            if term or trunc:
                (o0, o1), _ = env.reset()
    report("PuyoVersusEnv.step (with observations)", n, time.perf_counter() - t)


if __name__ == "__main__":
    bench_raw_game()
    bench_solo_env()
    bench_versus_env()
