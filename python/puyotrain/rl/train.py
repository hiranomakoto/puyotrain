"""MaskablePPO による1人プレイの学習。

出力(``<output_dir>/<run_name>-<日時>/``)::

    config.yaml, config.json      使った設定
    baseline.json                 ベースラインの評価結果
    metrics.jsonl                 チェックポイントごとの指標(1行1チェックポイント)
    checkpoints/step_XXXXXXXXXX.zip   重み(古いものも消さない)
    checkpoints/step_XXXXXXXXXX.json  その時点の指標・ベースライン比較・設定
    replays/*.json, videos/*.mp4  固定シードのリプレイと動画
    tb/                           TensorBoard
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
from .agents import PPOAgent, make_agent
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


def build_callbacks(cfg: dict, run_dir: Path, baselines: dict, t0: float):
    from stable_baselines3.common.callbacks import BaseCallback

    seeds = eval_seeds(cfg)
    every = int(cfg["checkpoint"]["every_steps"])
    chash = config_hash(cfg)

    class Checkpointer(BaseCallback):
        """一定ステップごとに保存・評価・動画生成。ロールアウトの統計も TensorBoard に出す。"""

        def __init__(self):
            super().__init__()
            self.next_save = every
            self.last_saved = -1

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
            if self.last_saved != self.num_timesteps:
                self.checkpoint()

        def checkpoint(self) -> None:
            step = self.num_timesteps
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
            if cfg["video"]["enabled"]:
                for s in cfg["video"]["seeds"]:
                    videos.append(str(write_video(agent, s, name, run_dir, cfg,
                                                  {"run": run_dir.name})))
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


def build_model(cfg: dict, device: str, tensorboard_log: str | None):
    """設定から並列環境と MaskablePPO を作る。戻り値: (model, env, total_timesteps)。"""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .features import PuyoFeatures

    env_cfg = cfg["env"]
    vec_cls = SubprocVecEnv if env_cfg["vec_env"] == "subproc" else DummyVecEnv
    env = make_vec_env(
        _make_env(cfg),
        n_envs=env_cfg["n_envs"],
        seed=cfg["seed"],
        vec_env_cls=vec_cls,
        monitor_kwargs={"info_keywords": ("total_score", "max_chain")},
    )
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


def train(cfg: dict, run_dir: Path | None = None) -> Path:
    t0 = time.time()
    run_dir = make_run_dir(cfg, run_dir)
    device = resolve_device(cfg["device"])
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({**cfg, "_resolved_device": device, "_config_hash": config_hash(cfg)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"run dir: {run_dir}  device: {device}", flush=True)

    # ベースライン(同じ固定シード)と、その動画
    seeds = eval_seeds(cfg)
    baselines_full = {}
    for bname in cfg["eval"]["baselines"]:
        baselines_full[bname] = evaluate(make_agent(bname), seeds, cfg["eval"]["max_steps"])
        print(f"[baseline] {bname}: mean_score={baselines_full[bname]['mean_score']:.1f} "
              f"mean_max_chain={baselines_full[bname]['mean_max_chain']:.2f}", flush=True)
        if cfg["video"]["enabled"]:
            for s in cfg["video"]["seeds"]:
                write_video(make_agent(bname), s, f"baseline_{bname}", run_dir, cfg, {})
    (run_dir / "baseline.json").write_text(json.dumps(
        {"eval_seeds": {"start": seeds[0], "count": len(seeds)},
         "max_steps": cfg["eval"]["max_steps"], "results": baselines_full},
        indent=2, ensure_ascii=False), encoding="utf-8")
    baselines = summarize_baselines(baselines_full)

    model, env, total = build_model(cfg, device, str(run_dir / "tb"))
    callback = build_callbacks(cfg, run_dir, baselines, t0)
    try:
        model.learn(total_timesteps=total, callback=callback, tb_log_name="ppo")
    finally:
        env.close()
    print(f"done in {time.time() - t0:.0f}s: {run_dir}", flush=True)
    return run_dir


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="MaskablePPO で1人プレイを学習する")
    p.add_argument("--config", type=Path, help="設定 YAML(既定: configs/solo_ppo.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="設定の上書き(例: --set ppo.total_timesteps=200000)")
    p.add_argument("--run-dir", type=Path, help="出力先を直接指定する")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    train(cfg, args.run_dir)


if __name__ == "__main__":
    main()
