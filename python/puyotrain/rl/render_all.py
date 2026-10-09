"""ランの全チェックポイント(とベースライン)の固定シード動画をまとめて作り直す。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import PPOAgent, make_agent
from .config import load_config
from .render import render_replay
from .replay import load_replay, record_solo, save_replay


def run_config(run_dir: Path) -> dict:
    p = run_dir / "config.yaml"
    return load_config(p) if p.exists() else load_config()


def render_all(run_dir: Path, seeds: list[int] | None = None, force: bool = False,
               rerecord: bool = False, baselines: bool = True) -> list[Path]:
    cfg = run_config(run_dir)
    vcfg = cfg["video"]
    seeds = seeds if seeds is not None else list(vcfg["seeds"])
    jobs: list[tuple[str, callable]] = []
    if baselines:
        for b in cfg["eval"]["baselines"]:
            jobs.append((f"baseline_{b}", lambda b=b: make_agent(b)))
    for ckpt in sorted((run_dir / "checkpoints").glob("step_*.zip")):
        jobs.append((ckpt.stem, lambda c=ckpt: PPOAgent(c)))

    made = []
    for label, factory in jobs:
        agent = None
        for s in seeds:
            rp = run_dir / "replays" / f"{label}_seed{s}.json"
            mp4 = run_dir / "videos" / f"{label}_seed{s}.mp4"
            if mp4.exists() and not force:
                continue
            if rp.exists() and not rerecord:
                rep = load_replay(rp)
            else:
                agent = agent or factory()
                rep = record_solo(agent, s, vcfg["max_moves"], extra_meta={"run": run_dir.name})
                save_replay(rep, rp)
            made.append(render_replay(rep, mp4, vcfg))
            print(mp4, flush=True)
    return made


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="全チェックポイントの動画を作り直す")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", help="既定: 設定の video.seeds")
    p.add_argument("--force", action="store_true", help="既存の動画も上書きする")
    p.add_argument("--rerecord", action="store_true",
                   help="リプレイも対局し直す(既定は既存のリプレイ JSON を再利用)")
    p.add_argument("--no-baselines", action="store_true")
    args = p.parse_args(argv)
    made = render_all(args.run_dir, args.seeds, args.force, args.rerecord, not args.no_baselines)
    print(json.dumps({"rendered": len(made)}))


if __name__ == "__main__":
    main()
