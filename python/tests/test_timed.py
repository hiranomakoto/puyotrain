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
        "pool.weights={greedy: 1, sync_versus: 1, past: 2}",
        f"pool.fixed={{sync_versus: {versus_ckpt}}}",
        f"eval.opponents=[greedy, sync_versus, {versus_ckpt}]",
        f"init_from={versus_ckpt}", "video.enabled=false", *extra,
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

    run = train(timed_cfg(tmp_path, versus_ckpt, "env.n_envs=2", "profile.every_iterations=1",
                          "env.reward=style", "env.drill={fraction: 0.5, moves: 10}"),
                tmp_path / "run")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert set(rows[-1]["metrics"]) == {"greedy", "sync_versus", "step_128"}
    for k in ("mean_seconds", "mean_finish_chain", "big_finish_rate", "counter_win_rate"):
        assert k in rows[-1]["metrics"]["greedy"]
    assert rows[-1]["drill"]["games"] == 1 and "rate_10" in rows[-1]["drill"]
    agent = PPOAgent(run / "checkpoints" / "step_0000000256.zip")
    assert agent.obs_kind == "timed"
    with pytest.raises(ValueError):
        agent.observe(_puyo.Versus(0), 0)
    out = capsys.readouterr().out
    assert "[profile] 1 iters" in out
    line = next(x for x in out.splitlines() if x.startswith("[profile] 1 iters"))
    for part in ("env: ", "observe=", "advance=", "greedy=", "py="):
        assert part in line
    resume(run, ["ppo.total_timesteps=384"])
    assert (run / "checkpoints" / "step_0000000384.zip").exists()


def test_display_shows_falling_pair_and_advance_until_stops_at_limit():
    g = TimedVersus(5)
    d = g.display(0)
    assert d["phase"] == "ready" and d["falling"] == [] and d["cleared"] is None
    g.act(0, 0)
    g.act(1, 5)
    g.advance_until(30)
    assert g.now == 30
    d = g.display(0)
    assert d["phase"] == "placing" and len(d["falling"]) == 2
    rows = sorted(r for r, _, _ in d["falling"])
    assert 0 < rows[0] < 11  # 出現位置(11)と着地位置(0)の間
    g.advance_until(1000)  # 操作待ちで止まる
    assert g.needs_action() == (True, True) and g.now == 54


def test_timed_replay_roundtrip_and_render(tmp_path):
    from puyotrain.rl.agents import RandomAgent
    from puyotrain.rl.render import render_replay, timed_scenes
    from puyotrain.rl.replay import load_replay, record_timed, save_replay

    rep = record_timed((GreedyAgent(0), RandomAgent(1)), 3)
    assert rep["mode"] == "timed" and rep["actions"][0][0] == 0
    path = save_replay(rep, tmp_path / "r.json")
    loaded = load_replay(path)
    cfg = {"fps": 30, "frames_per_move": 1, "frames_per_chain_stage": 1, "end_hold_frames": 2,
           "timed_speed": 20.0}
    scenes = list(timed_scenes(loaded, cfg))
    assert scenes[-1].players[0].banner in ("WIN", "LOSE", "DRAW")
    # 再生の最後の時刻・勝敗が記録と同じ
    assert scenes[-1].footer.strip() == f"{rep['meta']['frames'] / 60:5.1f} s".strip()
    winner = 0 if rep["meta"]["result"] == "p1_win" else 1
    assert scenes[-1].players[winner].banner == "WIN"
    out = render_replay(loaded, tmp_path / "r.mp4", cfg)
    assert out.exists() and out.stat().st_size > 0


def test_pool_with_several_fixed_opponents_and_initial_past(tmp_path, versus_ckpt):
    from puyotrain.rl.train_versus import build_pool, opponent_label, pool_fixed

    cfg = timed_cfg(tmp_path, versus_ckpt, f"pool.fixed={{a: {versus_ckpt}, b: {versus_ckpt}}}",
                    "pool.weights={greedy: 0, a: 1, b: 1, past: 1}",
                    f"pool.initial_past=[{versus_ckpt}]")
    assert set(pool_fixed(cfg)) == {"a", "b"}
    pool = build_pool(cfg, "cpu")
    assert set(pool.members) == {"greedy", "a", "b", "past"}
    assert len(pool.members["past"]) == 1
    cats = {pool.sample()[0] for _ in range(200)}
    assert cats == {"a", "b", "past"}  # 重み0の greedy は選ばない
    assert opponent_label("a") == "a" and opponent_label(str(versus_ckpt)) == "step_128"
    # 以前の形(pool.solo)も名前 solo の固定の相手として読む
    old = load_config("configs/versus_ppo.yaml", [f"pool.solo={versus_ckpt}"])
    assert pool_fixed(old) == {"solo": str(versus_ckpt)}


def test_style_rules():
    from puyotrain.rl.style import StyleConfig, finish_chain, is_counter_win, win_reward

    st = StyleConfig()
    assert win_reward(2, st) == pytest.approx(0.3)
    assert win_reward(3, st) == pytest.approx(0.3)
    assert win_reward(10, st) == pytest.approx(1.5)
    assert win_reward(14, st) == pytest.approx(1.5)
    assert win_reward(6, st) == pytest.approx(0.3 + 1.2 * 3 / 7)
    me = {"max_chain": 9, "max_chain_start": 900, "last_chain": 2, "last_chain_start": 1500}
    them = {"max_chain": 6, "max_chain_start": 600, "last_chain": 6, "last_chain_start": 600}
    assert finish_chain(me) == 9  # とどめの2連鎖ではなく最大の9連鎖
    assert is_counter_win(me, them, st)
    assert not is_counter_win(me, {**them, "max_chain": 4}, st)  # 相手の連鎖が小さい
    assert not is_counter_win({**me, "max_chain": 6}, them, st)  # 上回っていない
    assert not is_counter_win({**me, "max_chain_start": 500}, them, st)  # 先に撃っている
    with pytest.raises(KeyError):
        StyleConfig.from_dict({"nope": 1})


def test_timed_env_style_reward():
    from puyotrain.rl.style import StyleConfig
    from puyotrain.rl.timed_selfplay import TimedVersusVecEnv

    pool = OpponentPool({"greedy": 1}, None)
    env = TimedVersusVecEnv(3, pool, safe_mask=True, attack_coef=0.0, potential_coef=0.05,
                            gamma=0.99, anneal_steps=0, seed=3, reward="style",
                            style=StyleConfig())
    env.reset()
    rng = np.random.default_rng(3)
    episodes, terminal = [], []
    for _ in range(200):
        masks = np.stack(env.env_method("action_masks"))
        _, rewards, dones, infos = env.step(np.array([rng.choice(np.flatnonzero(m))
                                                      for m in masks]))
        for d, r, info in zip(dones, rewards, infos):
            if d:
                episodes.append(info["episode"])
                terminal.append(r)
    assert episodes
    for e, r in zip(episodes, terminal):
        assert "finish_chain" in e and "counter_win" in e
        if e["result"] == -1:
            assert r < -0.9  # 負けは −1(+ 小さな整形)
            assert e["finish_chain"] == 0
    with pytest.raises(ValueError):
        TimedVersusVecEnv(1, pool, True, 0, 0, 0.99, 0, reward="nope")


def test_videos_for_configured_opponents(tmp_path, versus_ckpt):
    from puyotrain.rl.train_versus import train

    run = train(timed_cfg(tmp_path, versus_ckpt, "env.n_envs=2", "ppo.total_timesteps=128",
                          "video.enabled=true", "video.opponents=[greedy, sync_versus]",
                          "video.timed_speed=60", "video.end_hold_frames=1"), tmp_path / "run")
    names = sorted(p.name for p in (run / "videos").glob("*.mp4"))
    assert names == [f"step_0000000128_vs_{o}_{t}_seed0.mp4"
                     for o in ("greedy", "sync_versus") for t in ("p1", "p2")]


def test_drill_reward_and_config():
    from puyotrain.rl.style import DrillConfig, drill_reward

    d = DrillConfig(fraction=0.5)
    assert drill_reward(3, d) == 0.0
    assert drill_reward(10, d) == 1.0
    assert drill_reward(13, d) == 1.0
    assert drill_reward(8, d) == pytest.approx((5 / 7) ** 2)
    with pytest.raises(KeyError):
        DrillConfig.from_dict({"nope": 1})


def test_drill_games_in_env():
    from puyotrain.rl.style import DrillConfig
    from puyotrain.rl.timed_selfplay import TimedVersusVecEnv

    pool = OpponentPool({"greedy": 1}, None)
    env = TimedVersusVecEnv(6, pool, safe_mask=True, attack_coef=0.0, potential_coef=0.0,
                            gamma=0.99, anneal_steps=0, seed=5, reward="style",
                            drill=DrillConfig(fraction=0.5, moves=12))
    env.reset()
    assert any(env.is_drill) and not all(env.is_drill)
    rng = np.random.default_rng(5)
    drills, versus = [], []
    for _ in range(150):
        masks = np.stack(env.env_method("action_masks"))
        _, _, dones, infos = env.step(np.array([rng.choice(np.flatnonzero(m)) for m in masks]))
        for d, info in zip(dones, infos):
            if d:
                (drills if info["episode"]["opponent"] == "drill" else versus).append(
                    info["episode"])
    assert drills and versus
    # 練習の対局は12手で終わる(窒息すればそれより前)
    assert all(e["l"] <= 12 for e in drills)
    assert any(e["l"] == 12 for e in drills)
    assert all(e["result"] in (0, -1) for e in drills)


def test_play_drills_reports_chains():
    from puyotrain.rl.match import play_drills
    from puyotrain.rl.style import DrillConfig

    r = play_drills(GreedyAgent(0), [0, 1, 2], DrillConfig(moves=20))
    assert r["games"] == 3
    assert 0 <= r["rate_10"] <= r["rate_8"] <= 1
    assert r["mean_max_chain"] >= 1  # greedy は20手のうちに何かしら消す
    g = TimedVersus(0, None, True)
    assert g.needs_action() == (True, False) and g.phases()[1] == "idle"


def test_greedy_opponent_in_timed_pool_is_greedy():
    pool = OpponentPool({"greedy": 1}, None)
    assert isinstance(pool.sample()[1], GreedyAgent)
