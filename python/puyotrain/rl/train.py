"""MaskablePPO による1人プレイの学習。

出力(``<output_dir>/<run_name>-<日時>/``)::

    config.yaml, config.json      使った設定
    baseline.json                 ベースラインの評価結果
    metrics.jsonl                 チェックポイントごとの指標(1行1チェックポイント)
    checkpoints/step_XXXXXXXXXX.zip   重み(古いものも消さない)
    checkpoints/step_XXXXXXXXXX.json  その時点の指標・ベースライン比較・設定
    replays/*.json, videos/*.mp4  固定シードのリプレイと動画
    tb/                           TensorBoard
    resume_history.jsonl          再開の履歴(再開したときだけ)

再開: ``--resume <run_dir>`` で最新のチェックポイントから続きを学習する。
``ppo.total_timesteps`` は「通算でここまで学習する」目標値。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ..envs import PuyoSoloEnv
from .agents import PPOAgent, checkpoint_step, make_agent
from .config import config_hash, eval_seeds, load_config
from .evaluate import evaluate
from .render import render_replay
from .replay import record_solo, save_replay


def resolve_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def make_run_dir(cfg: dict, run_dir: Path | None) -> Path:
    if run_dir is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(cfg["output_dir"]) / f"{cfg['run_name']}-{stamp}"
    if (run_dir / "checkpoints").exists() and any((run_dir / "checkpoints").iterdir()):
        raise FileExistsError(f"{run_dir} already has checkpoints")
    for sub in ("checkpoints", "replays", "videos", "tb"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def _make_env(cfg: dict):
    env_cfg = cfg["env"]

    def factory():
        return PuyoSoloEnv(max_steps=env_cfg["max_steps"], reward_scale=env_cfg["reward_scale"])

    return factory


def write_video(agent, seed: int, label: str, run_dir: Path, cfg: dict, extra_meta: dict) -> Path:
    vcfg = cfg["video"]
    rep = record_solo(agent, seed, vcfg["max_moves"], extra_meta=extra_meta)
    save_replay(rep, run_dir / "replays" / f"{label}_seed{seed}.json")
    return render_replay(rep, run_dir / "videos" / f"{label}_seed{seed}.mp4", vcfg)


def summarize_baselines(baselines: dict) -> dict:
    return {
        name: {"mean_score": r["mean_score"], "mean_max_chain": r["mean_max_chain"]}
        for name, r in baselines.items()
    }


def compare(metrics: dict, baselines: dict) -> dict:
    out = {}
    for name, b in baselines.items():
        out[f"score_vs_{name}"] = metrics["mean_score"] / b["mean_score"] if b["mean_score"] else None
        out[f"chain_vs_{name}"] = (
            metrics["mean_max_chain"] / b["mean_max_chain"] if b["mean_max_chain"] else None
        )
    return out


def build_callbacks(cfg: dict, run_dir: Path, baselines: dict, t0: float,
                    start_step: int = 0, extra_record: dict | None = None):
    from stable_baselines3.common.callbacks import BaseCallback

    seeds = eval_seeds(cfg)
    every = int(cfg["checkpoint"]["every_steps"])
    chash = config_hash(cfg)

    class Checkpointer(BaseCallback):
        """一定ステップごとに保存・評価・動画生成。ロールアウトの統計も TensorBoard に出す。"""

        def __init__(self):
            super().__init__()
            self.next_save = (start_step // every + 1) * every
            self.last_saved = start_step if start_step else -1
            self.last_had_videos = True

        def _on_rollout_end(self) -> None:
            buf = self.model.ep_info_buffer
            if buf:
                self.logger.record("rollout/ep_score_mean",
                                   float(np.mean([e["total_score"] for e in buf])))
                self.logger.record("rollout/ep_max_chain_mean",
                                   float(np.mean([e["max_chain"] for e in buf])))

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
                agent = PPOAgent(ckpt, deterministic=cfg["eval"]["deterministic"],
                                 model=self.model)
                videos = [write_video(agent, s, name, run_dir, cfg, {"run": run_dir.name}).name
                          for s in cfg["video"]["seeds"]]
                meta = ckpt.with_suffix(".json")
                record = json.loads(meta.read_text(encoding="utf-8"))
                record["videos"] = videos
                meta.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                encoding="utf-8")

        def checkpoint(self, videos_enabled: bool = True, final: bool = False) -> None:
            step = self.model.num_timesteps
            video_every = max(1, int(cfg["video"].get("every_checkpoints", 1)))
            videos_enabled = videos_enabled and (final or (step // every) % video_every == 0)
            name = f"step_{step:010d}"
            ckpt = run_dir / "checkpoints" / f"{name}.zip"
            self.model.save(str(ckpt))
            self.last_saved = step

            agent = PPOAgent(ckpt, deterministic=cfg["eval"]["deterministic"], model=self.model)
            metrics = evaluate(agent, seeds, cfg["eval"]["max_steps"])
            comparison = compare(metrics, baselines)
            for k in ("mean_score", "mean_max_chain", "mean_moves", "death_rate"):
                self.logger.record(f"eval/{k}", metrics[k])
            for k, v in comparison.items():
                if v is not None:
                    self.logger.record(f"eval/{k}", v)
            for bname, b in baselines.items():
                self.logger.record(f"baseline/{bname}_mean_score", b["mean_score"])
                self.logger.record(f"baseline/{bname}_mean_max_chain", b["mean_max_chain"])
            self.logger.dump(step)

            videos = []
            if cfg["video"]["enabled"] and videos_enabled:
                for s in cfg["video"]["seeds"]:
                    videos.append(str(write_video(agent, s, name, run_dir, cfg,
                                                  {"run": run_dir.name})))
            self.last_had_videos = bool(videos)
            record = {
                "step": step,
                "checkpoint": ckpt.name,
                "time": _dt.datetime.now().isoformat(timespec="seconds"),
                "elapsed_sec": round(time.time() - t0, 1),
                "eval_seeds": {"start": seeds[0], "count": len(seeds)},
                "metrics": metrics,
                "baselines": baselines,
                "comparison": comparison,
                "videos": [Path(v).name for v in videos],
                "config_hash": chash,
                "config": cfg,
                **(extra_record or {}),
            }
            (run_dir / "checkpoints" / f"{name}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                slim = {k: record[k] for k in ("step", "elapsed_sec", "metrics", "comparison")}
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")
            if self.verbose >= 0:
                print(f"[checkpoint] {name}: mean_score={metrics['mean_score']:.1f} "
                      f"mean_max_chain={metrics['mean_max_chain']:.2f} "
                      f"({metrics['seconds']:.1f}s eval)", flush=True)

    return Checkpointer()


def build_env(cfg: dict, seed: int):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_cfg = cfg["env"]
    vec_cls = SubprocVecEnv if env_cfg["vec_env"] == "subproc" else DummyVecEnv
    return make_vec_env(
        _make_env(cfg),
        n_envs=env_cfg["n_envs"],
        seed=seed,
        vec_env_cls=vec_cls,
        monitor_kwargs={"info_keywords": ("total_score", "max_chain")},
    )


def build_model(cfg: dict, device: str, tensorboard_log: str | None):
    """設定から並列環境と MaskablePPO を作る。戻り値: (model, env, total_timesteps)。"""
    from sb3_contrib import MaskablePPO

    from .features import PuyoFeatures

    env = build_env(cfg, cfg["seed"])
    pcfg = cfg["policy"]
    ppo = dict(cfg["ppo"])
    total = int(ppo.pop("total_timesteps"))
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": PuyoFeatures,
            "features_extractor_kwargs": {
                "cnn_channels": tuple(pcfg["cnn_channels"]),
                "features_dim": pcfg["features_dim"],
            },
            "net_arch": {"pi": list(pcfg["net_arch"]), "vf": list(pcfg["net_arch"])},
        },
        tensorboard_log=tensorboard_log,
        device=device,
        seed=cfg["seed"],
        verbose=0,
        **ppo,
    )
    return model, env, total


def write_config(cfg: dict, run_dir: Path, device: str) -> None:
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({**cfg, "_resolved_device": device, "_config_hash": config_hash(cfg)},
                   indent=2, ensure_ascii=False), encoding="utf-8")


def _baseline_key(cfg: dict) -> dict:
    return {"eval_seeds": cfg["eval"]["seeds"], "max_steps": cfg["eval"]["max_steps"],
            "names": list(cfg["eval"]["baselines"])}


def prepare_baselines(cfg: dict, run_dir: Path) -> dict:
    """ベースラインを評価して baseline.json に保存する。評価条件が同じ既存の結果があれば再利用する。"""
    path = run_dir / "baseline.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("key") == _baseline_key(cfg):
            return summarize_baselines(saved["results"])
    seeds = eval_seeds(cfg)
    results = {}
    for bname in cfg["eval"]["baselines"]:
        results[bname] = evaluate(make_agent(bname), seeds, cfg["eval"]["max_steps"])
        print(f"[baseline] {bname}: mean_score={results[bname]['mean_score']:.1f} "
              f"mean_max_chain={results[bname]['mean_max_chain']:.2f}", flush=True)
        if cfg["video"]["enabled"]:
            for s in cfg["video"]["seeds"]:
                if not (run_dir / "videos" / f"baseline_{bname}_seed{s}.mp4").exists():
                    write_video(make_agent(bname), s, f"baseline_{bname}", run_dir, cfg, {})
    path.write_text(json.dumps(
        {"key": _baseline_key(cfg),
         "eval_seeds": {"start": seeds[0], "count": len(seeds)},
         "max_steps": cfg["eval"]["max_steps"], "results": results},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return summarize_baselines(results)


def _learn(model, env, steps: int, callback, run_dir: Path, t0: float) -> None:
    """学習を回す。Ctrl+C で止めたときは、その時点の重みを保存してから終える。"""
    interrupted = False
    try:
        model.learn(total_timesteps=steps, callback=callback, tb_log_name="ppo",
                    reset_num_timesteps=False)
    except KeyboardInterrupt:
        interrupted = True
        if callback.last_saved != model.num_timesteps:
            print("\n[interrupt] saving checkpoint (videos skipped; use render_all later)...",
                  flush=True)
            callback.checkpoint(videos_enabled=False)
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001  子プロセスが先に終了している場合
            pass
    state = "interrupted" if interrupted else "done"
    print(f"{state} at step {model.num_timesteps:,} ({time.time() - t0:.0f}s): {run_dir}\n"
          f"resume with: python scripts/train.py --resume {run_dir}", flush=True)


def train(cfg: dict, run_dir: Path | None = None) -> Path:
    t0 = time.time()
    run_dir = make_run_dir(cfg, run_dir)
    device = resolve_device(cfg["device"])
    write_config(cfg, run_dir, device)
    print(f"run dir: {run_dir}  device: {device}", flush=True)

    baselines = prepare_baselines(cfg, run_dir)
    model, env, total = build_model(cfg, device, str(run_dir / "tb"))
    callback = build_callbacks(cfg, run_dir, baselines, t0)
    _learn(model, env, total, callback, run_dir, t0)
    return run_dir


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = [p for p in (run_dir / "checkpoints").glob("step_*.zip")
             if checkpoint_step(p) is not None]
    return max(ckpts, key=checkpoint_step) if ckpts else None


# 再開時に変えてはいけない設定(重みの形が変わる)
_FROZEN_SECTIONS = ("policy",)


def resume(run_dir: Path, overrides: list[str] | None = None,
           from_checkpoint: Path | None = None, new_run_dir: Path | None = None) -> Path:
    """チェックポイントから学習を再開する。

    - 既定では ``run_dir`` の最新チェックポイントから、同じディレクトリに続きを書く。
    - ``from_checkpoint`` に古いチェックポイントを指定する場合は、後のチェックポイントを
      上書きしないよう ``new_run_dir``(分岐先)が必要。
    - ``overrides`` で学習率・並列数・目標ステップ数などを変えられる(``policy`` は不可)。
    """
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
        src_baseline = run_dir / "baseline.json"
        if src_baseline.exists():
            (out_dir / "baseline.json").write_text(src_baseline.read_text(encoding="utf-8"),
                                                   encoding="utf-8")
    elif latest is not None and checkpoint_step(latest) > start:
        raise ValueError(
            f"{ckpt.name} is older than the latest checkpoint {latest.name}; continuing in "
            f"the same run would overwrite later checkpoints. Pass --run-dir to branch.")

    target = int(cfg["ppo"]["total_timesteps"])
    if target <= start:
        raise ValueError(
            f"checkpoint is already at step {start:,} >= ppo.total_timesteps={target:,}; "
            f"raise it, e.g. --set ppo.total_timesteps={start * 2}")

    # 経過時間は元のチェックポイントの記録から引き継ぐ
    prev_elapsed = 0.0
    meta_path = ckpt.with_suffix(".json")
    if meta_path.exists():
        prev_elapsed = float(json.loads(meta_path.read_text(encoding="utf-8")).get("elapsed_sec", 0))
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

    baselines = prepare_baselines(cfg, out_dir)
    # 同じ乱数で同じエピソードを繰り返さないよう、環境のシードは再開ステップでずらす
    env = build_env(cfg, cfg["seed"] + start)
    ppo = {k: v for k, v in cfg["ppo"].items() if k != "total_timesteps"}
    model = MaskablePPO.load(str(ckpt), env=env, device=device,
                             tensorboard_log=str(out_dir / "tb"), **ppo)
    model.num_timesteps = start
    callback = build_callbacks(cfg, out_dir, baselines, t0, start_step=start,
                               extra_record={"resumed_from": str(ckpt)})
    _learn(model, env, target - start, callback, out_dir, t0)
    return out_dir


def _stop_on_sigterm() -> None:
    """docker stop などの SIGTERM を Ctrl+C と同じ扱いにする(チェックポイントを保存して終わる)。"""
    import signal

    def handler(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):  # メインスレッド以外など
        pass


def main(argv=None) -> None:
    _stop_on_sigterm()
    p = argparse.ArgumentParser(description="MaskablePPO で1人プレイを学習する")
    p.add_argument("--config", type=Path, help="設定 YAML(既定: configs/solo_ppo.yaml)")
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
        cfg = load_config(args.config, args.overrides)
        train(cfg, args.run_dir)


if __name__ == "__main__":
    main()
