"""時間のモデルの学習(特徴量、TimedBatch、自己対戦の環境、重みの引き継ぎ、学習・再開)のテスト。"""

import json

import numpy as np
import pytest

from puyotrain import TimedVersus, _puyo

pytest.importorskip("yaml")
pytest.importorskip("sb3_contrib")
th = pytest.importorskip("torch")

from puyotrain.rl.agents import GreedyAgent, PPOAgent  # noqa: E402
from puyotrain.rl.config import load_config  # noqa: E402
from puyotrain.rl.selfplay import OpponentPool  # noqa: E402

TINY_PPO = ["ppo.n_steps=64", "ppo.batch_size=64", "ppo.n_epochs=1"]
KINDS = {"solo": (_puyo.NUM_ACTION_FEATURES, 0),
         "versus": (_puyo.NUM_VERSUS_ACTION_FEATURES, _puyo.NUM_VERSUS_INFO),
         "timed": (_puyo.NUM_TIMED_ACTION_FEATURES, _puyo.NUM_TIMED_INFO)}


@pytest.fixture(scope="module")
def versus_ckpt(tmp_path_factory):
    """同期ターン制の対戦用の小さな重み(1人用の小さな重みから引き継いで少しだけ学習)。"""
    from puyotrain.rl.train import train
    from puyotrain.rl.train_versus import train as train_versus

    root = tmp_path_factory.mktemp("versus")
    solo = train(load_config(overrides=[
        "env.n_envs=2", "ppo.total_timesteps=128", *TINY_PPO, "checkpoint.every_steps=128",
        "eval.seeds.count=1", "eval.max_steps=10", "eval.baselines=[random]",
        "video.enabled=false", "policy.action_features=true", "env.safe_mask=true",
    ]), root / "solo") / "checkpoints" / "step_0000000128.zip"
    run = train_versus(load_config("configs/versus_ppo.yaml", [
        "env.n_envs=2", "ppo.total_timesteps=128", *TINY_PPO, "checkpoint.every_steps=128",
        "eval.seeds.count=1", "eval.max_turns=20", "video.enabled=false",
        f"init_from={solo}", f"pool.solo={solo}",
    ]), root / "versus")
    ckpt = run / "checkpoints" / "step_0000000128.zip"
    # 小さなモデルは手ごとの点数がほぼ同点なので、比べやすいよう広げる
    agent = PPOAgent(ckpt)
    agent.model.policy.features_extractor.scorer[-1].weight.data.normal_(0, 1.0)
    agent.model.policy.features_extractor.opponent.weight.data.normal_(0, 0.1)
    agent.model.save(ckpt)
    return ckpt


def timed_cfg(tmp_path, versus_ckpt, *extra):
    return load_config("configs/versus_timed_ppo.yaml", [
        "env.n_envs=4", "ppo.total_timesteps=256", *TINY_PPO, "checkpoint.every_steps=128",
        "pool.snapshot_every=128", "eval.seeds.count=1",
        f"eval.opponents=[greedy, solo, {versus_ckpt}]",
        f"init_from={versus_ckpt}", f"pool.solo={versus_ckpt}", *extra,
    ])


def play_some(g: TimedVersus, rng, steps: int):
    for _ in range(steps):
        if g.is_over:
            break
        for p in (0, 1):
            if g.needs_action()[p]:
                g.act(p, int(rng.choice(np.flatnonzero(g.action_mask(p)))))
        g.advance()


def test_timed_observe_shapes_and_safe_mask():
    g = TimedVersus(1)
    play_some(g, np.random.default_rng(0), 10)
    for kind, (n_feat, n_info) in KINDS.items():
        for p in (0, 1):
            obs = g.observe(p, kind)
            assert obs["action_features"].shape == (22, n_feat)
            if n_info:
                assert obs["versus_info"].shape == (n_info,)
            else:
                assert "versus_info" not in obs
            if g.needs_action()[p]:
                np.testing.assert_array_equal(obs["action_mask"], g.safe_action_mask(p))
            assert obs["board"].shape == (2, 6, 13, 6)
    assert len(_puyo.TIMED_ACTION_FEATURE_NAMES) == _puyo.NUM_TIMED_ACTION_FEATURES
    assert len(_puyo.TIMED_INFO_NAMES) == _puyo.NUM_TIMED_INFO


def test_timed_batch_matches_single_games():
    rng = np.random.default_rng(1)
    singles = [TimedVersus(s) for s in range(5)]
    batch = _puyo.TimedBatch(5)
    for i in range(5):
        batch.reset(i, i)
    for _ in range(40):
        need = batch.needs_action()
        np.testing.assert_array_equal(need, [g.needs_action() for g in singles])
        if batch.is_over().all():
            break
        idx, players, acts = [], [], []
        for i, g in enumerate(singles):
            for p in (0, 1):
                if need[i, p]:
                    a = int(rng.choice(np.flatnonzero(g.action_mask(p))))
                    g.act(p, a)
                    idx.append(i), players.append(p), acts.append(a)
        if idx:
            batch.act(idx, players, acts)
        for g in singles:
            g.advance()
        batch.advance(list(range(5)))
        np.testing.assert_array_equal(batch.now(), [g.now for g in singles])
        for kind in KINDS:
            obs = batch.observe(list(range(5)) * 2, [0] * 5 + [1] * 5, kind)
            for row, (i, p) in enumerate([(i, 0) for i in range(5)] + [(i, 1) for i in range(5)]):
                single = singles[i].observe(p, kind)
                for k in single:
                    np.testing.assert_array_equal(obs[k][row], single[k])
    live = [i for i in range(5) if not singles[i].is_over]
    ready = [(i, p) for i in live for p in (0, 1) if singles[i].needs_action()[p]]
    if ready:
        got = batch.greedy([i for i, _ in ready], [p for _, p in ready], [7] * len(ready))
        assert got == [singles[i].greedy_action(p, 7) for i, p in ready]


def test_timed_env_plays_full_games():
    from puyotrain.rl.timed_selfplay import TimedVersusVecEnv

    pool = OpponentPool({"greedy": 1}, None)
    env = TimedVersusVecEnv(3, pool, safe_mask=True, attack_coef=0.01, potential_coef=0.05,
                            gamma=0.99, anneal_steps=10_000, seed=2)
    obs = env.reset()
    assert obs["action_features"].shape == (3, 22, _puyo.NUM_TIMED_ACTION_FEATURES)
    assert obs["versus_info"].shape == (3, _puyo.NUM_TIMED_INFO)
    rng = np.random.default_rng(2)
    episodes = []
    for _ in range(200):
        masks = np.stack(env.env_method("action_masks"))
        assert masks.any(axis=1).all()
        obs, rewards, dones, infos = env.step(np.array([rng.choice(np.flatnonzero(m))
                                                        for m in masks]))
        for d, info in zip(dones, infos):
            if d:
                assert "terminal_observation" in info
                episodes.append(info["episode"])
    assert episodes
    e = episodes[0]
    assert {"result", "opponent", "max_chain", "seconds"} <= set(e)
    assert -1 in [x["result"] for x in episodes]  # random な手では greedy に負ける


def test_transfer_from_versus_starts_as_the_same_policy(tmp_path, versus_ckpt):
    from puyotrain.rl.train_versus import build_env, build_model, build_pool, transfer_weights

    cfg = timed_cfg(tmp_path, versus_ckpt)
    model = build_model(cfg, build_env(cfg, build_pool(cfg, "cpu"), 0), "cpu", None)
    report = transfer_weights(model, versus_ckpt)
    assert any(n.endswith("scorer.0.weight") for n in report["widened"])
    assert any(n.endswith("opponent.weight") for n in report["widened"])
    assert any(n.startswith("value_net") for n in report["copied"])  # 対戦用からは価値も
    src = PPOAgent(versus_ckpt)
    dst = PPOAgent("unused.zip", model=model, safe_mask=True)
    assert src.obs_kind == "versus" and dst.obs_kind == "timed"
    g = TimedVersus(4)
    rng = np.random.default_rng(4)
    compared = 0
    for _ in range(30):
        if g.is_over:
            break
        for p in (0, 1):
            if not g.needs_action()[p]:
                continue
            pairs = []
            for agent in (src, dst):
                policy = agent.model.policy
                obs = {k: th.as_tensor(v[None]) for k, v in agent.observe(g, p).items()}
                with th.no_grad():
                    f = policy.extract_features(obs)
                    pairs.append(policy.action_net(policy.mlp_extractor.forward_actor(f))[0])
            assert th.allclose(pairs[0], pairs[1], atol=1e-4)
            compared += 1
            g.act(p, int(rng.choice(np.flatnonzero(g.action_mask(p)))))
        g.advance()
    assert compared > 10


def test_tiny_timed_training_and_resume(tmp_path, versus_ckpt, capsys):
    from puyotrain.rl.train_versus import resume, train

    run = train(timed_cfg(tmp_path, versus_ckpt, "env.n_envs=2", "profile.every_iterations=1"),
                tmp_path / "run")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows[-1]["metrics"]) == 3
    assert "mean_seconds" in rows[-1]["metrics"]["greedy"]
    agent = PPOAgent(run / "checkpoints" / "step_0000000256.zip")
    assert agent.obs_kind == "timed"
    with pytest.raises(ValueError):
        agent.observe(_puyo.Versus(0), 0)
    out = capsys.readouterr().out
    assert "[profile] 1 iters" in out
    resume(run, ["ppo.total_timesteps=384"])
    assert (run / "checkpoints" / "step_0000000384.zip").exists()


def test_greedy_opponent_in_timed_pool_is_greedy():
    pool = OpponentPool({"greedy": 1}, None)
    assert isinstance(pool.sample()[1], GreedyAgent)
