"""固定シードでの評価(平均得点・平均最大連鎖数)。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .._puyo import Game
from .agents import Agent, make_agent


# この連鎖数以上を「大連鎖」として、撃てたゲームの割合を数える
BIG_CHAIN = 10


def play_solo(agent: Agent, seed: int, max_steps: int) -> dict:
    """1局打つ。行動列も返す(リプレイに使える)。"""
    agent.reset(seed)
    g = Game(seed)
    actions: list[int] = []
    max_chain = 0
    max_potential = 0
    while not g.is_over and g.turn < max_steps:
        a = agent.act(g)
        _, chains, _, _ = g.step(a)
        actions.append(a)
        max_chain = max(max_chain, chains)
        max_potential = max(max_potential, g.potential_chain()[0])
    return {
        "seed": seed,
        "score": g.score,
        "max_chain": max_chain,
        "max_potential": max_potential,
        "moves": g.turn,
        "dead": g.is_over,
        "actions": actions,
    }


def play_solo_batched(agent, seeds: list[int], max_steps: int) -> list[dict]:
    """全シードを同時に1手ずつ進める(``act_batch`` を持つ決定的なエージェント用)。
    バッチ推論の浮動小数の丸め差を除けば、``play_solo`` を1局ずつ呼んだ場合と同じ。"""
    games = [Game(s) for s in seeds]
    actions: list[list[int]] = [[] for _ in seeds]
    max_chain = [0] * len(seeds)
    max_potential = [0] * len(seeds)
    safe = getattr(agent, "safe_mask", False)
    while True:
        active = [i for i, g in enumerate(games) if not g.is_over and g.turn < max_steps]
        if not active:
            break
        observations = [games[i].observation() for i in active]
        if safe:
            for i, o in zip(active, observations):
                o["action_mask"] = games[i].safe_action_mask()
        acts = agent.act_batch(observations)
        for i, a in zip(active, acts):
            _, chains, _, _ = games[i].step(a)
            actions[i].append(a)
            max_chain[i] = max(max_chain[i], chains)
            max_potential[i] = max(max_potential[i], games[i].potential_chain()[0])
    return [
        {"seed": s, "score": g.score, "max_chain": mc, "max_potential": mp, "moves": g.turn,
         "dead": g.is_over, "actions": acts}
        for s, g, mc, mp, acts in zip(seeds, games, max_chain, max_potential, actions)
    ]


def evaluate(agent: Agent, seeds: list[int], max_steps: int) -> dict:
    """複数シードで打ち、平均得点・平均最大連鎖数などを返す。"""
    t = time.perf_counter()
    if hasattr(agent, "act_batch") and getattr(agent, "deterministic", False):
        games = play_solo_batched(agent, seeds, max_steps)
    else:
        games = [play_solo(agent, s, max_steps) for s in seeds]
    scores = np.array([g["score"] for g in games], dtype=float)
    chains = np.array([g["max_chain"] for g in games], dtype=float)
    return {
        "episodes": len(games),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "median_score": float(np.median(scores)),
        "mean_max_chain": float(chains.mean()),
        "best_max_chain": int(chains.max()),
        "big_chain_rate": float(np.mean(chains >= BIG_CHAIN)),
        "mean_max_potential": float(np.mean([g["max_potential"] for g in games])),
        "mean_moves": float(np.mean([g["moves"] for g in games])),
        "death_rate": float(np.mean([g["dead"] for g in games])),
        "seconds": time.perf_counter() - t,
    }


def run_baselines(names: list[str], seeds: list[int], max_steps: int) -> dict:
    return {n: evaluate(make_agent(n), seeds, max_steps) for n in names}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="ベースライン(またはチェックポイント)を固定シードで評価する")
    p.add_argument("agents", nargs="*", default=["random", "greedy"],
                   help="random / greedy / チェックポイントのパス")
    p.add_argument("--seeds", type=int, default=100, help="シード数(0 から連番)")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--out", type=Path, help="結果を書く JSON")
    args = p.parse_args(argv)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    results = {a: evaluate(make_agent(a), seeds, args.max_steps) for a in args.agents}
    print(f"{'agent':<40} {'mean_score':>12} {'mean_max_chain':>15} {'mean_moves':>11} {'death':>6}")
    for name, r in results.items():
        print(f"{name:<40} {r['mean_score']:>12.1f} {r['mean_max_chain']:>15.2f} "
              f"{r['mean_moves']:>11.1f} {r['death_rate']:>6.0%}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seeds": seeds, "max_steps": args.max_steps, "results": results}
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
