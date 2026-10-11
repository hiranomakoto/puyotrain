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

from .._puyo import TimedVersus, Versus
from .agents import Agent, PPOAgent, make_agent
from .replay import record_versus, replay_name, save_replay


def _actions(agent: Agent, games: list[Versus], players: list[int]) -> list[int]:
    """複数局の手をまとめて求める(PPO の重みは1回の推論で)。"""
    if isinstance(agent, PPOAgent):
        return agent.act_batch([agent.observe(g, p) for g, p in zip(games, players)])
    return [agent.act(g, p) for g, p in zip(games, players)]


def play_games(a: Agent, b: Agent, seeds: list[int], max_turns: int) -> list[dict]:
    """A 対 B を各シード左右入れ替えて打ち、1局ごとの A 視点の記録を返す。
    全局を同時に1手ずつ進め、PPO の重みは全局分をまとめて推論する。"""
    specs = [(s, swapped) for s in seeds for swapped in (False, True)]
    games = [Versus(s) for s, _ in specs]
    a_side = [1 if swapped else 0 for _, swapped in specs]
    max_chain = [[0, 0] for _ in specs]
    a.reset(seeds[0] if seeds else 0)
    b.reset(seeds[0] if seeds else 0)
    while True:
        active = [i for i, g in enumerate(games) if not g.is_over and g.turn < max_turns]
        if not active:
            break
        act_a = _actions(a, [games[i] for i in active], [a_side[i] for i in active])
        act_b = _actions(b, [games[i] for i in active], [1 - a_side[i] for i in active])
        for i, x, y in zip(active, act_a, act_b):
            pair = [0, 0]
            pair[a_side[i]], pair[1 - a_side[i]] = x, y
            res = games[i].step(*pair)
            for p in (0, 1):
                max_chain[i][p] = max(max_chain[i][p], res[p]["chains"])
    records = []
    for (seed, swapped), g, mc, me in zip(specs, games, max_chain, a_side):
        if not g.is_over:
            outcome = "timeout"
        elif g.is_draw:
            outcome = "draw"
        else:
            outcome = "win" if g.winner == me else "loss"
        scores = g.scores()
        records.append({
            "seed": seed,
            "swapped": swapped,
            "outcome": outcome,
            "turns": g.turn,
            "max_chain": [mc[me], mc[1 - me]],
            "score": [scores[me], scores[1 - me]],
        })
    return records


def play_timed_games(a: Agent, b: Agent, seeds: list[int],
                     timing: dict | None = None) -> list[dict]:
    """``play_games`` の時間のモデル版(``TimedVersus``)。全局を同時に進め、各エージェントは
    自分の席が操作待ちの局の分だけまとめて手を決める。打ち切りは ``timing["max_frames"]``。"""
    specs = [(s, swapped) for s in seeds for swapped in (False, True)]
    games = [TimedVersus(s, timing) for s, _ in specs]
    a_side = [1 if swapped else 0 for _, swapped in specs]
    a.reset(seeds[0] if seeds else 0)
    b.reset(seeds[0] if seeds else 0)
    while True:
        active = [i for i, g in enumerate(games) if not g.is_over]
        if not active:
            break
        for agent, mine in ((a, True), (b, False)):
            todo = []
            for i in active:
                p = a_side[i] if mine else 1 - a_side[i]
                if games[i].needs_action()[p]:
                    todo.append((i, p))
            if todo:
                acts = _actions(agent, [games[i] for i, _ in todo], [p for _, p in todo])
                for (i, p), x in zip(todo, acts):
                    games[i].act(p, x)
        for i in active:
            games[i].advance()
    records = []
    for (seed, swapped), g, me in zip(specs, games, a_side):
        if g.timed_out:
            outcome = "timeout"
        elif g.is_draw:
            outcome = "draw"
        else:
            outcome = "win" if g.winner == me else "loss"
        scores, chains, moves = g.scores(), g.max_chains(), g.moves()
        records.append({
            "seed": seed,
            "swapped": swapped,
            "outcome": outcome,
            "turns": moves[me],
            "frames": g.now,
            "max_chain": [chains[me], chains[1 - me]],
            "score": [scores[me], scores[1 - me]],
        })
    return records


def play_match(a: Agent, b: Agent, seeds: list[int], max_turns: int,
               replay_dir: Path | None = None, n_replays: int = 0,
               timed: bool = False, timing: dict | None = None) -> dict:
    """A 対 B を各シード左右入れ替えて打ち、A 視点の集計を返す。
    ``replay_dir`` を指定すると、最初の ``n_replays`` シード分の対局をリプレイとして保存する。
    ``timed`` なら時間のモデル(``TimedVersus``)で打つ(``max_turns`` は使わず、打ち切りは
    ``timing["max_frames"]``。リプレイは未対応)。"""
    t = time.perf_counter()
    if timed:
        if replay_dir is not None:
            raise ValueError("replays are not supported for timed games yet")
        games = play_timed_games(a, b, seeds, timing)
    else:
        games = play_games(a, b, seeds, max_turns)
    if replay_dir is not None:
        for seed in seeds[:n_replays]:
            for pair in ((a, b), (b, a)):
                rep = record_versus(pair, seed, max_turns)
                save_replay(rep, replay_dir / f"{replay_name(rep['meta']['labels'], seed)}.json")
    outcomes = [g["outcome"] for g in games]
    n = len(games)
    decided = sum(o in ("win", "loss") for o in outcomes)
    wins = outcomes.count("win")
    extra = {"mean_seconds": float(np.mean([g["frames"] for g in games])) / 60} if timed else {}
    return {
        **extra,
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
    p.add_argument("--timed", action="store_true",
                   help="時間のモデル(TimedVersus)で打つ。時間の定数は既定値")
    args = p.parse_args(argv)

    a = make_agent(args.agent_a, seed=0, device=args.device)
    b = make_agent(args.agent_b, seed=1, device=args.device)
    replay_dir = (args.out.parent if args.out else Path(".")) / "replays"
    r = play_match(a, b, list(range(args.seeds)), args.max_turns,
                   replay_dir if args.replays else None, args.replays, timed=args.timed)
    rate = r["win_rate_decided"]
    length = f"time={r['mean_seconds']:.1f}s" if args.timed else ""
    print(f"{args.agent_a} vs {args.agent_b}: {r['wins']}W {r['losses']}L "
          f"{r['draws']}D {r['timeouts']}T / {r['games']} games  "
          f"win_rate={r['win_rate']:.3f} (decided {rate if rate is None else round(rate, 3)})  "
          f"turns={r['mean_turns']:.1f} {length} max_chain={r['mean_max_chain'][0]:.2f}"
          f"/{r['mean_max_chain'][1]:.2f}  ({r['seconds']:.0f}s)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"agent_a": args.agent_a, "agent_b": args.agent_b,
                   "max_turns": args.max_turns, **r}
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
