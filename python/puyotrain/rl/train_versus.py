"""MaskablePPO による対戦(自己対戦)の学習。

出力(``<output_dir>/<run_name>-<日時>/``)::

    config.yaml, config.json      使った設定
    metrics.jsonl                 チェックポイントごとの評価(固定の相手との勝率)
    checkpoints/step_XXXXXXXXXX.zip / .json   重みとその時点の評価
    pool.json                     相手のプールに入っている過去の自分(確認用)
    replays/*.json, videos/*.mp4  greedy との対局のリプレイと動画
    tb/                           TensorBoard
    resume_history.jsonl          再開の履歴(再開したときだけ)

学習の始め方: ``init_from`` の1人プレイの重みから、形が同じ部分をコピーして始める
(``transfer_solo_weights``)。追加した部分は影響ゼロで始まるので、学習開始時点の方策は
1人プレイの重みと同じ。価値ヘッドは目的(得点 → 勝敗)が違うので引き継がない。

再開: ``--resume <run_dir>`` で最新のチェックポイントから続きを学習する。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import numpy as np

from .._puyo import NUM_ACTION_FEATURES
from .agents import Agent, GreedyAgent, PPOAgent, checkpoint_step
from .config import config_hash, eval_seeds, load_config
from .match import play_match
from .render import render_replay
from .replay import record_versus, save_replay
from .selfplay import OpponentPool, VersusVecEnv
from .train import (
    _learn,
    _stop_on_sigterm,
    latest_checkpoint,
    make_run_dir,
    resolve_device,
    write_config,
)

DEFAULT_VERSUS_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "versus_ppo.yaml"
SCRIPT = "scripts/train_versus.py"


# ---------------------------------------------------------------------- モデル


def transfer_solo_weights(model, solo_path: str | Path) -> dict:
    """1人プレイの重み(``policy.action_features``)を対戦用のモデルにコピーする。

    - 形が同じパラメータはそのままコピーする。
    - 手ごとの点数づけの1層目は、入力が「特徴量13個 + 文脈」から「特徴量20個 + 文脈」に
      増えるので、1人用の13個と文脈の部分だけコピーし、増えた7個の列は0にする。
    - 相手の情報を足す ``opponent`` 層は1人用に無いので、0のまま(影響ゼロ)。
    - 価値ヘッド(``mlp_extractor.value_net``、``value_net``)は引き継がない。

    戻り値: {"copied": [...], "widened": [...], "skipped": [...]}(パラメータ名)
    """
    import torch as th
    from sb3_contrib import MaskablePPO

    src = MaskablePPO.load(str(solo_path), device="cpu").policy.state_dict()
    dst = model.policy.state_dict()
    report: dict[str, list[str]] = {"copied": [], "widened": [], "skipped": []}
    for name, param in dst.items():
        s = src.get(name)
        if name.startswith(("mlp_extractor.value_net", "value_net")) or s is None:
            report["skipped"].append(name)
        elif s.shape == param.shape:
            dst[name] = s.clone()
            report["copied"].append(name)
        elif name.endswith("scorer.0.weight") and s.shape[0] == param.shape[0]:
            n_ctx = s.shape[1] - NUM_ACTION_FEATURES
            w = th.zeros_like(param)
            w[:, :NUM_ACTION_FEATURES] = s[:, :NUM_ACTION_FEATURES]
            w[:, param.shape[1] - n_ctx:] = s[:, NUM_ACTION_FEATURES:]
            dst[name] = w
            report["widened"].append(name)
        else:
            report["skipped"].append(name)
    model.policy.load_state_dict(dst)
    return report


def build_pool(cfg: dict, device: str, snapshots: list[str] | None = None) -> OpponentPool:
    pcfg = cfg["pool"]
    pool = OpponentPool(pcfg["weights"], pcfg["solo"], device=device,
                        max_snapshots=int(pcfg["max_snapshots"]), seed=cfg["seed"])
    for p in snapshots or []:
        pool.add_snapshot(Path(p))
    return pool


def build_env(cfg: dict, pool: OpponentPool, seed: int, start_step: int = 0) -> VersusVecEnv:
    e = cfg["env"]
    return VersusVecEnv(
        n_envs=int(e["n_envs"]), pool=pool, max_turns=int(e["max_turns"]),
        safe_mask=bool(e["safe_mask"]), attack_coef=float(e["attack_coef"]),
        potential_coef=float(e["potential_coef"]), gamma=float(cfg["ppo"]["gamma"]),
        anneal_steps=int(e["anneal_steps"]), seed=seed, start_step=start_step,
    )


def build_model(cfg: dict, env: VersusVecEnv, device: str, tensorboard_log: str | None):
    from sb3_contrib import MaskablePPO

    from .features import PuyoActionPolicy, PuyoVersusFeatures

    pcfg = cfg["policy"]
    ppo = {k: v for k, v in cfg["ppo"].items() if k != "total_timesteps"}
    return MaskablePPO(
        PuyoActionPolicy,
        env,
        policy_kwargs={
            "features_extractor_class": PuyoVersusFeatures,
            "features_extractor_kwargs": {
                "cnn_channels": tuple(pcfg["cnn_channels"]),
                "features_dim": pcfg["features_dim"],
                "piece_planes": pcfg["piece_planes"],
                "action_hidden": tuple(pcfg["action_hidden"]),
            },
            "net_arch": {"pi": [], "vf": list(pcfg["net_arch"])},
        },
        tensorboard_log=tensorboard_log,
        device=device,
        seed=cfg["seed"],
        verbose=0,
        **ppo,
    )


# ---------------------------------------------------------------------- 評価・記録


def eval_opponent(spec: str, cfg: dict, device: str) -> Agent:
    if spec == "greedy":
        return GreedyAgent(1)
    if spec == "solo":
        return PPOAgent(cfg["pool"]["solo"], device=device)
    return PPOAgent(spec, device=device)


def opponent_label(spec: str) -> str:
    if spec in ("greedy", "solo"):
        return spec
    step = checkpoint_step(spec)
    return f"step_{step}" if step is not None else Path(spec).stem


def build_callbacks(cfg: dict, run_dir: Path, pool: OpponentPool, env: VersusVecEnv,
                    device: str, t0: float, start_step: int = 0,
                    extra_record: dict | None = None):
    from stable_baselines3.common.callbacks import BaseCallback

    seeds = eval_seeds(cfg)
    every = int(cfg["checkpoint"]["every_steps"])
    snap_every = int(cfg["pool"]["snapshot_every"])
    chash = config_hash(cfg)
    opponents = {opponent_label(s): eval_opponent(s, cfg, device) for s in cfg["eval"]["opponents"]}

    class Checkpointer(BaseCallback):
        """一定ステップごとに保存・評価・動画。過去の自分をプールに加える。勝率を TensorBoard に出す。"""

        def __init__(self):
            super().__init__()
            self.next_save = (start_step // every + 1) * every
            self.next_snapshot = (start_step // snap_every + 1) * snap_every
            self.last_saved = start_step if start_step else -1
            self.last_had_videos = True

        def _on_rollout_end(self) -> None:
            buf = self.model.ep_info_buffer
            if buf:
                results = [e["result"] for e in buf]
                self.logger.record("rollout/win_rate", float(np.mean([r == 1 for r in results])))
                for cat in ("greedy", "solo", "past"):
                    rs = [e["result"] for e in buf if e["opponent"] == cat]
                    if rs:
                        self.logger.record(f"rollout/win_rate_{cat}",
                                           float(np.mean([r == 1 for r in rs])))
            self.logger.record("train/aux_scale", env.anneal())
            self.logger.record("train/pool_past", len(pool.members["past"]))

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_save:
                self.checkpoint()
                while self.next_save <= self.num_timesteps:
                    self.next_save += every
            return True

        def _on_training_end(self) -> None:
            if self.last_saved != self.model.num_timesteps:
                self.checkpoint(final=True)
            elif not self.last_had_videos and cfg["video"]["enabled"]:
                # 最後のチェックポイントが間引きで動画なしだった場合も、終了時の重みの動画は作る
                name = f"step_{self.last_saved:010d}"
                ckpt = run_dir / "checkpoints" / f"{name}.zip"
                me = PPOAgent(ckpt, model=self.model, safe_mask=cfg["env"]["safe_mask"])
                meta = ckpt.with_suffix(".json")
                record = json.loads(meta.read_text(encoding="utf-8"))
                record["videos"] = self.write_videos(me, name)
                meta.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                encoding="utf-8")

        def write_videos(self, me: Agent, name: str) -> list[str]:
            greedy = GreedyAgent(1)
            videos = []
            for s in cfg["video"]["seeds"]:
                for agents, tag in (((me, greedy), "p1"), ((greedy, me), "p2")):
                    rep = record_versus(agents, s, int(cfg["video"]["max_moves"]),
                                        extra_meta={"run": run_dir.name})
                    stem = f"{name}_vs_greedy_{tag}_seed{s}"
                    save_replay(rep, run_dir / "replays" / f"{stem}.json")
                    videos.append(render_replay(rep, run_dir / "videos" / f"{stem}.mp4",
                                                cfg["video"]).name)
            return videos

        def checkpoint(self, videos_enabled: bool = True, final: bool = False) -> None:
            step = self.model.num_timesteps
            name = f"step_{step:010d}"
            ckpt = run_dir / "checkpoints" / f"{name}.zip"
            self.model.save(str(ckpt))
            self.last_saved = step

            if step >= self.next_snapshot:
                pool.add_snapshot(ckpt)
                (run_dir / "pool.json").write_text(
                    json.dumps({"past": pool.snapshot_paths()}, indent=2), encoding="utf-8")
                while self.next_snapshot <= step:
                    self.next_snapshot += snap_every

            me = PPOAgent(ckpt, model=self.model, safe_mask=cfg["env"]["safe_mask"])
            metrics = {}
            for label, opp in opponents.items():
                r = play_match(me, opp, seeds, int(cfg["eval"]["max_turns"]))
                r.pop("per_game")
                metrics[label] = r
                self.logger.record(f"eval/win_rate_vs_{label}", r["win_rate"])
                self.logger.record(f"eval/turns_vs_{label}", r["mean_turns"])
                self.logger.record(f"eval/max_chain_vs_{label}", r["mean_max_chain"][0])
            self.logger.dump(step)

            video_every = max(1, int(cfg["video"].get("every_checkpoints", 1)))
            videos = []
            if (cfg["video"]["enabled"] and videos_enabled
                    and (final or (step // every) % video_every == 0)):
                videos = self.write_videos(me, name)
            self.last_had_videos = bool(videos)

            record = {
                "step": step,
                "checkpoint": ckpt.name,
                "time": _dt.datetime.now().isoformat(timespec="seconds"),
                "elapsed_sec": round(time.time() - t0, 1),
                "eval_seeds": {"start": seeds[0], "count": len(seeds)},
                "metrics": metrics,
                "aux_scale": env.anneal(),
                "pool_past": pool.snapshot_paths(),
                "videos": videos,
                "config_hash": chash,
                "config": cfg,
                **(extra_record or {}),
            }
            (run_dir / "checkpoints" / f"{name}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                slim = {k: record[k] for k in ("step", "elapsed_sec", "metrics", "aux_scale")}
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")
            if self.verbose >= 0:
                rates = "  ".join(f"vs_{k}={m['win_rate']:.2f}" for k, m in metrics.items())
                print(f"[checkpoint] {name}: {rates}  aux={env.anneal():.2f}", flush=True)

    return Checkpointer()


# ---------------------------------------------------------------------- 学習・再開


def train(cfg: dict, run_dir: Path | None = None) -> Path:
    t0 = time.time()
    run_dir = make_run_dir(cfg, run_dir)
    device = resolve_device(cfg["device"])
    write_config(cfg, run_dir, device)
    print(f"run dir: {run_dir}  device: {device}", flush=True)

    pool = build_pool(cfg, device)
    env = build_env(cfg, pool, cfg["seed"])
    model = build_model(cfg, env, device, str(run_dir / "tb"))
    if cfg.get("init_from"):
        rep = transfer_solo_weights(model, cfg["init_from"])
        print(f"init_from {cfg['init_from']}: copied {len(rep['copied'])}, "
              f"widened {len(rep['widened'])}, fresh {len(rep['skipped'])} tensors", flush=True)
    callback = build_callbacks(cfg, run_dir, pool, env, device, t0)
    _learn(model, env, int(cfg["ppo"]["total_timesteps"]), callback, run_dir, t0, SCRIPT)
    return run_dir


# 再開時に変えてはいけない設定(重みの形が変わる)
_FROZEN_SECTIONS = ("policy",)


def resume(run_dir: Path, overrides: list[str] | None = None,
           from_checkpoint: Path | None = None, new_run_dir: Path | None = None) -> Path:
    """チェックポイントから学習を再開する(``train.resume`` と同じ使い方)。
    相手のプール(過去の自分)は、そのチェックポイントの記録(.json の ``pool_past``)から戻す。"""
    from sb3_contrib import MaskablePPO

    run_dir = Path(run_dir)
    old_cfg = load_config(run_dir / "config.yaml")
    cfg = load_config(run_dir / "config.yaml", overrides)
    for sec in _FROZEN_SECTIONS:
        if cfg[sec] != old_cfg[sec]:
            raise ValueError(f"'{sec}' cannot change on resume (network shape would differ)")

    latest = latest_checkpoint(run_dir)
    ckpt = Path(from_checkpoint) if from_checkpoint else latest
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint to resume from in {run_dir}")
    start = checkpoint_step(ckpt)
    if start is None:
        raise ValueError(f"cannot read the step number from {ckpt.name}")

    out_dir = run_dir
    if new_run_dir is not None and Path(new_run_dir).resolve() != run_dir.resolve():
        out_dir = make_run_dir(cfg, Path(new_run_dir))
    elif latest is not None and checkpoint_step(latest) > start:
        raise ValueError(
            f"{ckpt.name} is older than the latest checkpoint {latest.name}; continuing in "
            f"the same run would overwrite later checkpoints. Pass --run-dir to branch.")

    target = int(cfg["ppo"]["total_timesteps"])
    if target <= start:
        raise ValueError(
            f"checkpoint is already at step {start:,} >= ppo.total_timesteps={target:,}; "
            f"raise it, e.g. --set ppo.total_timesteps={start * 2}")

    prev_elapsed = 0.0
    meta_path = ckpt.with_suffix(".json")
    snapshots: list[str] = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prev_elapsed = float(meta.get("elapsed_sec", 0))
        snapshots = list(meta.get("pool_past", []))
    t0 = time.time() - prev_elapsed

    device = resolve_device(cfg["device"])
    write_config(cfg, out_dir, device)
    with open(out_dir / "resume_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": _dt.datetime.now().isoformat(timespec="seconds"),
            "from_checkpoint": str(ckpt), "from_step": start, "target_step": target,
            "overrides": overrides or [], "config_hash": config_hash(cfg),
        }, ensure_ascii=False) + "\n")
    print(f"resume: {ckpt} (step {start:,}) -> {out_dir}  target {target:,}  device: {device}",
          flush=True)

    pool = build_pool(cfg, device, [p for p in snapshots if Path(p).exists()])
    # 同じ乱数で同じ対局を繰り返さないよう、環境のシードは再開ステップでずらす
    env = build_env(cfg, pool, cfg["seed"] + start, start_step=start)
    ppo = {k: v for k, v in cfg["ppo"].items() if k != "total_timesteps"}
    model = MaskablePPO.load(str(ckpt), env=env, device=device,
                             tensorboard_log=str(out_dir / "tb"), **ppo)
    model.num_timesteps = start
    callback = build_callbacks(cfg, out_dir, pool, env, device, t0, start_step=start,
                               extra_record={"resumed_from": str(ckpt)})
    _learn(model, env, target - start, callback, out_dir, t0, SCRIPT)
    return out_dir


def main(argv=None) -> None:
    _stop_on_sigterm()
    p = argparse.ArgumentParser(description="MaskablePPO で対戦(自己対戦)を学習する")
    p.add_argument("--config", type=Path, help="設定 YAML(既定: configs/versus_ppo.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="設定の上書き(例: --set ppo.total_timesteps=200000)")
    p.add_argument("--run-dir", type=Path,
                   help="出力先を直接指定する(--resume と併用すると分岐先)")
    p.add_argument("--resume", type=Path, metavar="RUN_DIR",
                   help="このランの最新チェックポイントから再開する")
    p.add_argument("--from", dest="from_checkpoint", type=Path, metavar="CHECKPOINT",
                   help="--resume で使うチェックポイントを指定する(古いものなら --run-dir が必要)")
    args = p.parse_args(argv)
    if args.resume:
        if args.config:
            p.error("--resume uses the run's own config.yaml; change settings with --set")
        resume(args.resume, args.overrides, args.from_checkpoint, args.run_dir)
    else:
        if args.from_checkpoint:
            p.error("--from requires --resume")
        cfg = load_config(args.config or DEFAULT_VERSUS_CONFIG, args.overrides)
        train(cfg, args.run_dir)


if __name__ == "__main__":
    main()
