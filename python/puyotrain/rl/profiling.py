"""学習ループのプロファイリング。

1イテレーション(ロールアウト収集 + ネットワーク更新)ごとに、所要時間を次の区分に分けて測る。

- env: 環境シミュレーション(Rust の対局進行、特徴量の計算、greedy の相手、観測の組み立て)。
  環境が内訳(``env.<名前>`` の区分)を記録していれば、env の中の割合も出す
  (時間のモデルの環境: observe = 観測・特徴量、advance = 手の決定と時刻の進行、greedy = greedy の手、
  py = 残りの Python 側の処理)
- opp_infer: 相手の推論(自己対戦で、相手が PPO の重みのとき)
- infer: 学習側の推論(ロールアウト中の方策・価値の順伝播)
- transfer: CPU → GPU の転送(ロールアウト中の観測、更新時のミニバッチ)
- update: ネットワーク更新(順伝播・逆伝播・最適化。transfer を除く)
- checkpoint: 保存・評価・動画(チェックポイントのイテレーションだけ)
- other: 残り(バッファへの追加、行動の GPU → CPU、コールバックなど)

GPU の処理は非同期なので、計測区間の終わりで ``torch.cuda.synchronize()`` を呼ぶ
(その分だけ少し遅くなる)。``every`` イテレーションごとに合計して割合をログに出し、
TensorBoard の ``perf/*`` にも記録する。

- steps/sec: 環境ステップ数 ÷ イテレーション時間(学習全体の速さ)
- samples/sec: 更新で処理したサンプル数(バッファの大きさ × n_epochs)÷ update 時間
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

SECTIONS = ("env", "opp_infer", "infer", "transfer", "update", "checkpoint", "other")


class SectionTimer:
    """区分ごとの合計時間。``section`` は入れ子にでき、外側の時間から内側の時間を引けるよう
    ``nested`` に内側の合計も記録する。"""

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.totals: dict[str, float] = defaultdict(float)

    def sync(self) -> None:
        if self.cuda:
            import torch

            torch.cuda.synchronize()

    @contextmanager
    def section(self, name: str, sync: bool = False):
        if sync:
            self.sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                self.sync()
            self.totals[name] += time.perf_counter() - t

    def wrap(self, fn, name: str, sync: bool = False):
        @wraps(fn)
        def timed(*args, **kwargs):
            with self.section(name, sync):
                return fn(*args, **kwargs)

        return timed

    def take(self) -> dict[str, float]:
        out = dict(self.totals)
        self.totals.clear()
        return out


def count_parameters(model) -> dict[str, int]:
    """方策ネットのパラメータ数(全体と主な部品ごと)。共有している部品は1回だけ数える。"""
    policy = model.policy
    seen: set[int] = set()

    def count(module) -> int:
        n = 0
        for p in module.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
        return n

    parts = {"features_extractor": count(policy.features_extractor),
             "mlp_extractor": count(policy.mlp_extractor),
             "action_net": count(policy.action_net),
             "value_net": count(policy.value_net)}
    parts["total"] = sum(p.numel() for p in policy.parameters())
    return parts


def describe_setup(model, env) -> str:
    """起動時に表示する、並列環境数・バッチサイズ・パラメータ数・GPU の情報。"""
    import torch

    params = count_parameters(model)
    lines = [
        f"[profile] n_envs={env.num_envs} n_steps={model.n_steps} "
        f"buffer={env.num_envs * model.n_steps} batch_size={model.batch_size} "
        f"n_epochs={model.n_epochs} device={model.device}",
        "[profile] parameters: " + ", ".join(f"{k}={v:,}" for k, v in params.items()),
    ]
    if model.device.type == "cuda":
        props = torch.cuda.get_device_properties(model.device)
        lines.append(f"[profile] gpu: {props.name}, {props.total_memory / 2**30:.1f} GiB")
    return "\n".join(lines)


def attach_profiler(model, env, every: int = 10):
    """モデルと環境に計測を仕込み、ログを出すコールバックを返す。

    ``checkpointer`` の ``checkpoint`` メソッドを計測したいときは、返したコールバックの
    ``watch_checkpointer(cb)`` を呼ぶ。
    """
    import sb3_contrib.ppo_mask.ppo_mask as ppo_mask_module
    from stable_baselines3.common.callbacks import BaseCallback

    timer = SectionTimer(cuda=model.device.type == "cuda")

    # 環境: VecEnv.step は step_async → step_wait。自己対戦の環境は相手の推論を内側で測る
    env.step_wait = timer.wrap(env.step_wait, "env_total")
    if hasattr(env, "timer"):
        env.timer = timer
    # 学習側の推論(ロールアウト中の forward)と CPU → GPU の転送
    model.policy.forward = timer.wrap(model.policy.forward, "infer", sync=True)
    ppo_mask_module.obs_as_tensor = timer.wrap(ppo_mask_module.obs_as_tensor, "transfer_rollout",
                                               sync=True)
    # 更新(中のミニバッチ転送は別に測って引く)
    model.train = timer.wrap(model.train, "train_total", sync=True)
    # 差し替えた関数はモデルの保存(model.save)の対象から外す
    excluded = model._excluded_save_params
    model._excluded_save_params = lambda: [*excluded(), "train", "_excluded_save_params"]
    buffer = model.rollout_buffer
    buffer.to_torch = timer.wrap(buffer.to_torch, "transfer_update", sync=True)

    class ProfileCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.iter_start: float | None = None
            self.acc: dict[str, float] = defaultdict(float)
            self.iters = 0
            self.steps = 0
            self.env_parts: dict[str, float] = defaultdict(float)

        def watch_checkpointer(self, cb) -> None:
            if hasattr(cb, "checkpoint"):
                cb.checkpoint = timer.wrap(cb.checkpoint, "checkpoint")

        def _on_training_start(self) -> None:
            print(describe_setup(self.model, self.training_env), flush=True)

        def _on_rollout_start(self) -> None:
            now = time.perf_counter()
            if self.iter_start is not None:
                self._finish_iteration(now - self.iter_start)
            self.iter_start = now

        def _on_step(self) -> bool:
            return True

        def _finish_iteration(self, wall: float) -> None:
            t = timer.take()
            opp = t.get("opp_infer", 0.0)
            env_t = t.get("env_total", 0.0) - opp
            transfer = t.get("transfer_rollout", 0.0) + t.get("transfer_update", 0.0)
            update = t.get("train_total", 0.0) - t.get("transfer_update", 0.0)
            ckpt = t.get("checkpoint", 0.0)
            parts = {"env": env_t, "opp_infer": opp, "infer": t.get("infer", 0.0),
                     "transfer": transfer, "update": update, "checkpoint": ckpt}
            parts["other"] = max(0.0, wall - sum(parts.values()))
            for k, v in parts.items():
                self.acc[k] += v
            inner = {k[4:]: v for k, v in t.items() if k.startswith("env.")}
            if inner:
                inner["py"] = max(0.0, env_t - sum(inner.values()))
                for k, v in inner.items():
                    self.env_parts[k] += v
            self.acc["wall"] += wall
            self.acc["update_time"] += update
            self.iters += 1
            self.steps += self.model.n_steps * self.training_env.num_envs
            if self.iters >= every:
                self._log()

        def _log(self) -> None:
            import torch

            wall = self.acc["wall"]
            buffer = self.model.n_steps * self.training_env.num_envs
            samples = buffer * self.model.n_epochs * self.iters
            sps = self.steps / wall
            sample_rate = samples / self.acc["update_time"] if self.acc["update_time"] else 0.0
            for k in SECTIONS:
                self.logger.record(f"perf/frac_{k}", self.acc[k] / wall)
            self.logger.record("perf/steps_per_sec", sps)
            self.logger.record("perf/samples_per_sec", sample_rate)
            mem = ""
            if self.model.device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(self.model.device) / 2**20
                self.logger.record("perf/gpu_peak_mib", peak)
                mem = f"  gpu_peak={peak:.0f}MiB"
            fracs = " ".join(f"{k}={100 * self.acc[k] / wall:.1f}%" for k in SECTIONS)
            inner = ""
            if self.env_parts:
                for k, v in self.env_parts.items():
                    self.logger.record(f"perf/frac_env_{k}", v / wall)
                inner = "  env: " + " ".join(f"{k}={100 * v / wall:.1f}%"
                                             for k, v in self.env_parts.items())
            print(f"[profile] {self.iters} iters {wall:.1f}s  steps/s={sps:.0f} "
                  f"samples/s={sample_rate:.0f}  {fracs}{mem}{inner}", flush=True)
            self.acc.clear()
            self.env_parts.clear()
            self.iters = 0
            self.steps = 0

    return ProfileCallback()
