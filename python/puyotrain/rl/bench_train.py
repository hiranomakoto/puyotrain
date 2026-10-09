"""学習速度(1秒あたりの環境ステップ数)を CPU / GPU で比べる。評価・動画・保存はしない。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import load_config
from .train import build_model, resolve_device


def bench(cfg: dict, device: str, steps: int) -> float:
    model, env, _ = build_model(cfg, device, None)
    try:
        # 1回目の更新までを暖機として除く
        warm = cfg["env"]["n_envs"] * cfg["ppo"]["n_steps"]
        model.learn(total_timesteps=warm)
        t = time.perf_counter()
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        return steps / (time.perf_counter() - t)
    finally:
        env.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="学習速度を device ごとに計測する")
    p.add_argument("--config", type=Path)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--devices", nargs="+", default=["cpu", "auto"],
                   help="cpu / cuda / auto(auto は GPU があれば cuda)")
    p.add_argument("--steps", type=int, default=30000)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seen = set()
    for d in args.devices:
        dev = resolve_device(d)
        if dev in seen:
            continue
        seen.add(dev)
        fps = bench(cfg, dev, args.steps)
        print(f"{dev:<6} {fps:>10,.0f} env steps/s  (n_envs={cfg['env']['n_envs']}, "
              f"vec_env={cfg['env']['vec_env']})", flush=True)


if __name__ == "__main__":
    main()
