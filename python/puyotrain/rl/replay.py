"""対局してリプレイ JSON を保存する。

リプレイ形式(決定的なのでシードと行動列だけで完全に再現できる)::

    {
      "format": "puyotrain-replay", "version": 1,
      "mode": "solo" | "versus",
      "seed": 0,
      "actions": [3, 12, ...]            # solo
               | [[3, 5], [12, 0], ...], # versus
      "meta": {"agents": [...], "labels": [...], "training_steps": [...],
               "final_scores": [...], "max_chains": [...], "moves": 123,
               "result": "dead" | "max_moves" | "p1_win" | "p2_win" | "draw",
               "max_moves": 500, "created_at": "..."}
    }
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Optional

from .._puyo import Versus
from .agents import Agent, agent_label, make_agent
from .evaluate import play_solo

FORMAT = "puyotrain-replay"
VERSION = 1


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _training_step(agent: Agent) -> Optional[int]:
    return agent.describe().get("step")


def record_solo(agent: Agent, seed: int, max_moves: int, extra_meta: Optional[dict] = None) -> dict:
    r = play_solo(agent, seed, max_moves)
    meta = {
        "agents": [agent.describe()],
        "labels": [agent_label(agent)],
        "training_steps": [_training_step(agent)],
        "final_scores": [r["score"]],
        "max_chains": [r["max_chain"]],
        "moves": r["moves"],
        "result": "dead" if r["dead"] else "max_moves",
        "max_moves": max_moves,
        "created_at": _now(),
    }
    meta.update(extra_meta or {})
    return {
        "format": FORMAT,
        "version": VERSION,
        "mode": "solo",
        "seed": seed,
        "actions": r["actions"],
        "meta": meta,
    }


def record_versus(
    agents: tuple[Agent, Agent], seed: int, max_moves: int, extra_meta: Optional[dict] = None
) -> dict:
    for a in agents:
        a.reset(seed)
    v = Versus(seed)
    actions: list[list[int]] = []
    max_chains = [0, 0]
    while not v.is_over and v.turn < max_moves:
        a0 = agents[0].act(v, 0)
        a1 = agents[1].act(v, 1)
        res = v.step(a0, a1)
        actions.append([a0, a1])
        for i in range(2):
            max_chains[i] = max(max_chains[i], res[i]["chains"])
    if not v.is_over:
        result = "max_moves"
    elif v.is_draw:
        result = "draw"
    else:
        result = f"p{v.winner + 1}_win"
    meta = {
        "agents": [a.describe() for a in agents],
        "labels": [agent_label(a) for a in agents],
        "training_steps": [_training_step(a) for a in agents],
        "final_scores": list(v.scores()),
        "max_chains": max_chains,
        "moves": v.turn,
        "result": result,
        "max_moves": max_moves,
        "created_at": _now(),
    }
    meta.update(extra_meta or {})
    return {
        "format": FORMAT,
        "version": VERSION,
        "mode": "versus",
        "seed": seed,
        "actions": actions,
        "meta": meta,
    }


def save_replay(replay: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")
    return path


def load_replay(path: Path) -> dict:
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    if r.get("format") != FORMAT:
        raise ValueError(f"{path} is not a puyotrain replay")
    if r.get("version") != VERSION:
        raise ValueError(f"unsupported replay version: {r.get('version')}")
    return r


def replay_name(labels: list[str], seed: int) -> str:
    return f"{'_vs_'.join(labels)}_seed{seed}"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="チェックポイント(または random/greedy)で対局してリプレイを保存する")
    p.add_argument("--agent", required=True, help="random / greedy / チェックポイント(.zip)")
    p.add_argument("--agent2", help="指定すると対戦(左 = --agent、右 = --agent2)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-moves", type=int, default=500)
    p.add_argument("--out-dir", type=Path, default=Path("replays"))
    p.add_argument("--render", action="store_true", help="続けて MP4 も作る")
    p.add_argument("--config", type=Path, help="動画設定を読む YAML(既定: configs/solo_ppo.yaml)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    a1 = make_agent(args.agent, device=args.device)
    a2 = make_agent(args.agent2, seed=1, device=args.device) if args.agent2 else None
    for seed in args.seeds:
        if a2 is None:
            rep = record_solo(a1, seed, args.max_moves)
        else:
            rep = record_versus((a1, a2), seed, args.max_moves)
        path = save_replay(rep, args.out_dir / f"{replay_name(rep['meta']['labels'], seed)}.json")
        m = rep["meta"]
        print(f"{path}: moves={m['moves']} scores={m['final_scores']} "
              f"max_chains={m['max_chains']} result={m['result']}")
        if args.render:
            from .config import load_config
            from .render import render_replay

            video = load_config(args.config)["video"]
            out = render_replay(rep, path.with_suffix(".mp4"), video)
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
