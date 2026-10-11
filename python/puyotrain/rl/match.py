"""2つのエージェントを固定シードで対戦させ、勝率を測る。

各シードで左右を入れ替えて2局打つ(席の違い = おじゃまの降り方の乱数の違いを打ち消すため)。
決着がつかずに ``max_turns`` に達した局は「打ち切り」として、勝ちにも負けにも数えない。

例::

    python scripts/match.py runs/solo_afterstate_b/checkpoints/step_0019000000.zip greedy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .agents import Agent, make_agent
from .replay import record_versus, replay_name, save_replay


def _game_record(rep: dict, swapped: bool) -> dict:
    """リプレイのメタ情報を、1つ目のエージェント(A)視点の記録にする。"""
    m = rep["meta"]
    a, b = (1, 0) if swapped else (0, 1)
    result = m["result"]
    if result == f"p{a + 1}_win":
        outcome = "win"
    elif result == f"p{b + 1}_win":
        outcome = "loss"
    elif result == "draw":
        outcome = "draw"
    else:
        outcome = "timeout"
    return {
        "seed": rep["seed"],
        "swapped": swapped,
        "outcome": outcome,
        "turns": m["moves"],
        "max_chain": [m["max_chains"][a], m["max_chains"][b]],
        "score": [m["final_scores"][a], m["final_scores"][b]],
    }


def play_match(a: Agent, b: Agent, seeds: list[int], max_turns: int,
               replay_dir: Path | None = None, n_replays: int = 0) -> dict:
    """A 対 B を各シード左右入れ替えて打ち、A 視点の集計を返す。"""
    t = time.perf_counter()
    games = []
    for i, seed in enumerate(seeds):
        for swapped in (False, True):
            pair = (b, a) if swapped else (a, b)
            rep = record_versus(pair, seed, max_turns)
            games.append(_game_record(rep, swapped))
            if replay_dir is not None and i < n_replays:
                save_replay(rep, replay_dir / f"{replay_name(rep['meta']['labels'], seed)}.json")
    outcomes = [g["outcome"] for g in games]
    n = len(games)
    decided = sum(o in ("win", "loss") for o in outcomes)
    wins = outcomes.count("win")
    return {
        "games": n,
        "wins": wins,
        "losses": outcomes.count("loss"),
        "draws": outcomes.count("draw"),
        "timeouts": outcomes.count("timeout"),
        "win_rate": wins / n,
        "win_rate_decided": wins / decided if decided else None,
        "mean_turns": float(np.mean([g["turns"] for g in games])),
        "mean_max_chain": [float(np.mean([g["max_chain"][k] for g in games])) for k in (0, 1)],
        "seconds": time.perf_counter() - t,
        "per_game": games,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="2つのエージェントを対戦させて勝率を測る")
    p.add_argument("agent_a", help="random / greedy / チェックポイント(.zip)")
    p.add_argument("agent_b", help="random / greedy / チェックポイント(.zip)")
    p.add_argument("--seeds", type=int, default=100, help="シード数(0から。各シード2局)")
    p.add_argument("--max-turns", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, help="結果を JSON で保存する")
    p.add_argument("--replays", type=int, default=0,
                   help="最初の N シード分のリプレイを保存する(--out と同じフォルダの replays/)")
    args = p.parse_args(argv)

    a = make_agent(args.agent_a, seed=0, device=args.device)
    b = make_agent(args.agent_b, seed=1, device=args.device)
    replay_dir = (args.out.parent if args.out else Path(".")) / "replays"
    r = play_match(a, b, list(range(args.seeds)), args.max_turns,
                   replay_dir if args.replays else None, args.replays)
    rate = r["win_rate_decided"]
    print(f"{args.agent_a} vs {args.agent_b}: {r['wins']}W {r['losses']}L "
          f"{r['draws']}D {r['timeouts']}T / {r['games']} games  "
          f"win_rate={r['win_rate']:.3f} (decided {rate if rate is None else round(rate, 3)})  "
          f"turns={r['mean_turns']:.1f}  max_chain={r['mean_max_chain'][0]:.2f}"
          f"/{r['mean_max_chain'][1]:.2f}  ({r['seconds']:.0f}s)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"agent_a": args.agent_a, "agent_b": args.agent_b,
                   "max_turns": args.max_turns, **r}
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
