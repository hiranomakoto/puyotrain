"""チェックポイントどうしの総当たりで、強さを Elo レーティングにまとめる。

自己対戦では「A は B に強く、B は C に強く、C は A に強い」のような関係が起きうるので、
1つ前のチェックポイントとの勝率だけでは強くなっているか判断できない。総当たりの結果から
Bradley-Terry モデル(Elo と同じ勝率の式)で全員の強さを1本の数直線に並べる。

- 勝率の式: P(i が j に勝つ) = 1 / (1 + 10^((R_j − R_i) / 400))
- 引き分けと打ち切りは0.5勝として数える。全勝・全敗でも発散しないよう、
  対戦したペアごとに仮想の引き分けを1局足す。
- greedy がいれば greedy を 1000 にそろえる(いなければ最初の参加者)。
- 対戦結果は JSON に保存し、次回は新しいペアだけを打つ(シード数・最大ターンが同じ場合)。

例::

    python scripts/league.py runs/versus_main --every 2500000 --include greedy solo
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .agents import Agent, GreedyAgent, PPOAgent, RandomAgent, checkpoint_step
from .match import play_match


def fit_elo(players: list[str], results: dict[str, dict], anchor: str | None = None,
            iterations: int = 2000) -> dict[str, float]:
    """対戦結果(``"A|B"`` → A 視点の wins / losses / draws / timeouts)から Elo を推定する。"""
    idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    wins = np.zeros((n, n))  # wins[i, j] = i が j から取った勝ち数(引き分けは0.5)
    games = np.zeros((n, n))
    for key, r in results.items():
        a, b = key.split("|")
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        half = 0.5 * (r["draws"] + r["timeouts"]) + 0.5  # 仮想の引き分け1局
        total = r["wins"] + r["losses"] + r["draws"] + r["timeouts"] + 1
        wins[i, j] += r["wins"] + half
        wins[j, i] += r["losses"] + half
        games[i, j] += total
        games[j, i] += total
    # Bradley-Terry の MM 法(Hunter 2004)
    p = np.ones(n)
    for _ in range(iterations):
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1)
        new = np.where(denom > 0, wins.sum(axis=1) / np.maximum(denom, 1e-12), p)
        new /= np.exp(np.mean(np.log(np.maximum(new, 1e-12))))
        if np.allclose(new, p, rtol=1e-10, atol=0):
            p = new
            break
        p = new
    ratings = 400 * np.log10(np.maximum(p, 1e-12))
    ref = idx[anchor] if anchor in idx else 0
    ratings += 1000 - ratings[ref]
    return {pl: float(r) for pl, r in zip(players, ratings)}


def expected_score(ra: float, rb: float) -> float:
    """Elo の差から予想される勝率(A 視点)。"""
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def run_players(run_dir: Path, every: int) -> list[Path]:
    """ラン内のチェックポイントのうち、``every`` の倍数のステップと最新のもの。"""
    ckpts = sorted((p for p in (run_dir / "checkpoints").glob("step_*.zip")
                    if checkpoint_step(p) is not None), key=checkpoint_step)
    picked = [p for p in ckpts if checkpoint_step(p) % every == 0]
    if ckpts and ckpts[-1] not in picked:
        picked.append(ckpts[-1])
    return picked


def label_for(path: Path, run_dir: Path | None) -> str:
    step = checkpoint_step(path)
    name = f"{step / 1e6:g}M" if step is not None else path.stem
    if run_dir is not None and path.parent.parent.resolve() == run_dir.resolve():
        return name
    return f"{path.parent.parent.name}:{name}"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="チェックポイントの総当たりで Elo レーティングを出す")
    p.add_argument("run", type=Path, nargs="?", help="ランのフォルダ(チェックポイントを参加者にする)")
    p.add_argument("--every", type=int, default=2_500_000,
                   help="この倍数のステップのチェックポイントと最新のものを参加させる")
    p.add_argument("--include", nargs="*", default=["greedy"],
                   help="greedy / random / solo(--solo の重み)/ チェックポイントのパス")
    p.add_argument("--solo", default="runs/solo_afterstate_b/checkpoints/step_0019000000.zip")
    p.add_argument("--seeds", type=int, default=25, help="1ペアあたりのシード数(各2局)")
    p.add_argument("--max-turns", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, help="結果の JSON(既定: <run>/league.json)")
    args = p.parse_args(argv)
    if args.run is None and not args.include:
        p.error("give a run directory or --include players")

    agents: dict[str, object] = {}  # ラベル → エージェントを作る関数
    for spec in args.include:
        if spec == "greedy":
            agents["greedy"] = lambda: GreedyAgent(1)
        elif spec == "random":
            agents["random"] = lambda: RandomAgent(1)
        elif spec == "solo":
            agents["solo"] = lambda: PPOAgent(args.solo, device=args.device)
        else:
            path = Path(spec)
            agents[label_for(path, args.run)] = lambda path=path: PPOAgent(path, device=args.device)
    if args.run is not None:
        for path in run_players(args.run, args.every):
            agents[label_for(path, args.run)] = lambda path=path: PPOAgent(path, device=args.device)
    players = list(agents)

    out = args.out or ((args.run or Path(".")) / "league.json")
    key = {"seeds": args.seeds, "max_turns": args.max_turns}
    results: dict[str, dict] = {}
    if out.exists():
        saved = json.loads(out.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            results = saved["results"]

    loaded: dict[str, Agent] = {}

    def agent(label: str) -> Agent:
        if label not in loaded:
            loaded[label] = agents[label]()
        return loaded[label]

    seeds = list(range(args.seeds))
    todo = [(a, b) for a, b in itertools.combinations(players, 2)
            if f"{a}|{b}" not in results and f"{b}|{a}" not in results]
    print(f"{len(players)} players, {len(todo)} new pairs × {2 * args.seeds} games", flush=True)
    t0 = time.perf_counter()
    for k, (a, b) in enumerate(todo, 1):
        r = play_match(agent(a), agent(b), seeds, args.max_turns)
        results[f"{a}|{b}"] = {x: r[x] for x in ("wins", "losses", "draws", "timeouts")}
        print(f"  [{k}/{len(todo)}] {a} vs {b}: {r['wins']}-{r['losses']}"
              f" ({r['seconds']:.0f}s)", flush=True)

    anchor = "greedy" if "greedy" in players else None
    ratings = fit_elo(players, results, anchor)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"key": key, "players": players, "ratings": ratings,
                               "results": results}, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print(f"\nElo ({'greedy' if anchor else players[0]} = 1000, {time.perf_counter() - t0:.0f}s):")
    for pl in players:
        print(f"  {pl:>24}  {ratings[pl]:7.0f}")
    best = max(players, key=ratings.get)
    print(f"best: {best}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
