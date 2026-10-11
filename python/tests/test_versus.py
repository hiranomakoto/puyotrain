"""対戦の学習(対戦用の特徴量、自己対戦の環境、1人用の重みの引き継ぎ、学習・再開)のテスト。"""

import json

import numpy as np
import pytest

from puyotrain import Versus, _puyo
from puyotrain.envs import versus_observation

pytest.importorskip("yaml")
pytest.importorskip("sb3_contrib")
th = pytest.importorskip("torch")

from puyotrain.rl.agents import GreedyAgent, PPOAgent  # noqa: E402
from puyotrain.rl.config import load_config  # noqa: E402
from puyotrain.rl.selfplay import OpponentPool, VersusVecEnv  # noqa: E402

TINY_PPO = ["ppo.n_steps=64", "ppo.batch_size=64", "ppo.n_epochs=1"]


@pytest.fixture(scope="module")
def solo_ckpt(tmp_path_factory):
    """手ごとの特徴量を使う1人用の小さな重み。"""
    from puyotrain.rl.train import train

    cfg = load_config(overrides=[
        "env.n_envs=2", "ppo.total_timesteps=128", *TINY_PPO, "checkpoint.every_steps=128",
        "eval.seeds.count=1", "eval.max_steps=10", "eval.baselines=[random]",
        "video.enabled=false", "policy.action_features=true", "env.safe_mask=true",
    ])
    run = train(cfg, tmp_path_factory.mktemp("solo") / "run")
    return run / "checkpoints" / "step_0000000128.zip"


def versus_cfg(tmp_path, solo_ckpt, *extra):
    return load_config("configs/versus_ppo.yaml", [
        "env.n_envs=2", "ppo.total_timesteps=256", *TINY_PPO, "checkpoint.every_steps=128",
        "pool.snapshot_every=128", "eval.seeds.count=1", "eval.max_turns=40",
        "video.seeds=[0]", "video.max_moves=8", "video.frames_per_move=1",
        "video.frames_per_chain_stage=1", "video.end_hold_frames=1",
        f"init_from={solo_ckpt}", f"pool.solo={solo_ckpt}", *extra,
    ])


def test_versus_observation_features():
    v = Versus(3)
    rng = np.random.default_rng(0)
    for _ in range(15):
        for p in (0, 1):
            obs = versus_observation(v, p, safe_mask=True, versus_features=True)
            f = obs["action_features"]
            assert f.shape == (22, _puyo.NUM_VERSUS_ACTION_FEATURES)
            np.testing.assert_array_equal(f[:, :_puyo.NUM_ACTION_FEATURES], v.action_features(p))
            assert obs["versus_info"].shape == (_puyo.NUM_VERSUS_INFO,)
            np.testing.assert_array_equal(obs["action_mask"], v.safe_action_mask(p))
        v.step(*[int(rng.choice(np.flatnonzero(v.action_mask(p)))) for p in (0, 1)])
    assert len(_puyo.VERSUS_ACTION_FEATURE_NAMES) == _puyo.NUM_VERSUS_ACTION_FEATURES
    assert len(_puyo.VERSUS_INFO_NAMES) == _puyo.NUM_VERSUS_INFO


def test_versus_vec_env_plays_full_games():
    pool = OpponentPool({"greedy": 1, "solo": 1, "past": 2}, None)
    env = VersusVecEnv(3, pool, max_turns=200, safe_mask=True, attack_coef=0.01,
                       potential_coef=0.05, gamma=0.99, anneal_steps=300, seed=1)
    obs = env.reset()
    assert obs["action_features"].shape == (3, 22, _puyo.NUM_VERSUS_ACTION_FEATURES)
    assert env.has_attr("action_masks")
    rng = np.random.default_rng(1)
    episodes = []
    for _ in range(120):
        masks = np.stack(env.env_method("action_masks"))
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        obs, rewards, dones, infos = env.step(acts)
        assert rewards.shape == (3,)
        for d, info in zip(dones, infos):
            if d:
                assert "terminal_observation" in info
                episodes.append(info["episode"])
    assert episodes
    assert all(e["opponent"] == "greedy" for e in episodes)  # solo / past は空
    assert all(e["result"] in (-1, 0, 1) for e in episodes)
    assert -1 in [e["result"] for e in episodes]  # random な手では greedy に負ける
    assert env.anneal() == 0.0  # 120手 × 3局 > 300


def test_transfer_starts_as_solo_policy(tmp_path, solo_ckpt):
    from puyotrain.rl.train_versus import build_env, build_model, build_pool, transfer_solo_weights

    cfg = versus_cfg(tmp_path, solo_ckpt)
    pool = build_pool(cfg, "cpu")
    model = build_model(cfg, build_env(cfg, pool, 0), "cpu", None)
    report = transfer_solo_weights(model, solo_ckpt)
    assert any(n.endswith("scorer.0.weight") for n in report["widened"])
    assert any("opponent" in n for n in report["skipped"])

    solo = PPOAgent(solo_ckpt)
    v = Versus(5)
    rng = np.random.default_rng(2)
    for _ in range(12):
        # 予告おじゃまがあっても、引き継いだ直後は1人用と同じ方策
        for p in (0, 1):
            obs_s, _ = solo.model.policy.obs_to_tensor(
                {k: x[None] for k, x in solo.observe(v, p).items()})
            obs_v, _ = model.policy.obs_to_tensor(
                {k: x[None] for k, x in versus_observation(v, p, True, versus_features=True)
                 .items()})
            with th.no_grad():
                ls = solo.model.policy.get_distribution(obs_s).distribution.logits
                lv = model.policy.get_distribution(obs_v).distribution.logits
            assert th.allclose(ls, lv, atol=1e-5)
        v.step(*[int(rng.choice(np.flatnonzero(v.action_mask(p)))) for p in (0, 1)])


def test_tiny_versus_training_and_resume(tmp_path, solo_ckpt):
    from puyotrain.rl.train_versus import resume, train

    run = train(versus_cfg(tmp_path, solo_ckpt), tmp_path / "run")
    ckpts = sorted(p.name for p in (run / "checkpoints").glob("step_*.zip"))
    assert ckpts == ["step_0000000128.zip", "step_0000000256.zip"]
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert set(rows[-1]["metrics"]) == {"greedy", "solo"}
    meta = json.loads((run / "checkpoints" / "step_0000000256.json").read_text())
    assert len(meta["pool_past"]) == 2  # 128 ステップごとに過去の自分を追加
    assert list((run / "videos").glob("*.mp4"))

    agent = PPOAgent(run / "checkpoints" / "step_0000000256.zip")
    assert agent.versus_features
    with pytest.raises(ValueError):
        agent.observe(_puyo.Game(0))

    resume(run, ["ppo.total_timesteps=384"])
    assert (run / "checkpoints" / "step_0000000384.zip").exists()
    assert "resume_history.jsonl" in [p.name for p in run.iterdir()]


def test_greedy_pool_member_resets_per_game():
    pool = OpponentPool({"greedy": 1}, None)
    cat, agent = pool.sample()
    assert cat == "greedy" and isinstance(agent, GreedyAgent)
