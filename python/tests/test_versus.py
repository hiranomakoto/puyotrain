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


def test_play_games_batched_matches_sequential_for_deterministic_agents(solo_ckpt):
    from puyotrain.rl.match import play_games
    from puyotrain.rl.replay import record_versus

    a, b = PPOAgent(solo_ckpt), PPOAgent(solo_ckpt)
    batched = play_games(a, b, [0, 1], 60)
    for g in batched:
        pair = (b, a) if g["swapped"] else (a, b)
        m = record_versus(pair, g["seed"], 60)["meta"]
        assert m["moves"] == g["turns"]


def test_fast_act_batch_matches_predict(tmp_path, solo_ckpt):
    """act_batch の高速な道(直接 argmax)は、SB3 の predict(決定的)と同じ手を返す。"""
    from puyotrain.rl.train_versus import build_env, build_model, build_pool, transfer_solo_weights

    cfg = versus_cfg(tmp_path, solo_ckpt)
    model = build_model(cfg, build_env(cfg, build_pool(cfg, "cpu"), 0), "cpu", None)
    transfer_solo_weights(model, solo_ckpt)
    model.policy.features_extractor.opponent.weight.data.normal_(0, 0.1)  # 相手の情報も効かせる
    for agent in (PPOAgent(solo_ckpt), PPOAgent("unused.zip", model=model)):
        v = Versus(9)
        rng = np.random.default_rng(3)
        for _ in range(25):
            obs = [agent.observe(v, p) for p in (0, 1)]
            stacked = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
            expected, _ = agent.model.predict(stacked, action_masks=stacked["action_mask"]
                                              .astype(bool), deterministic=True)
            assert agent.act_batch(obs) == [int(a) for a in expected]
            if v.is_over:
                break
            v.step(*[int(rng.choice(np.flatnonzero(v.action_mask(p)))) for p in (0, 1)])


def test_stacked_past_opponents_match_individual_inference(tmp_path, solo_ckpt):
    """過去の自分を束ねた推論で選んだ手は、1人ずつ推論したときの最高点の手(同点を含む)。

    置いた結果が同じになる手どうしは特徴量が同じで点数も同点になり、同点のどれを選ぶかは
    計算順の違い(1e-5 程度)で変わるので、手そのものではなく点数で比べる。
    """
    from puyotrain.rl.train_versus import build_env, build_model, build_pool, transfer_solo_weights

    cfg = versus_cfg(tmp_path, solo_ckpt)
    paths = []
    for k in range(3):
        model = build_model(cfg, build_env(cfg, build_pool(cfg, "cpu"), 0), "cpu", None)
        transfer_solo_weights(model, solo_ckpt)
        th.manual_seed(k)
        model.policy.features_extractor.opponent.weight.data.normal_(0, 0.1)  # 3つとも違う重み
        model.policy.features_extractor.scorer[-1].weight.data.normal_(0, 1.0)
        paths.append(tmp_path / f"step_{k}.zip")
        model.save(paths[-1])
    pool = OpponentPool({"past": 1}, None, device="cpu")
    for p in paths:
        pool.add_snapshot(p)
    env = VersusVecEnv(12, pool, max_turns=200, safe_mask=True, attack_coef=0.0,
                       potential_coef=0.0, gamma=0.99, anneal_steps=0, seed=4)
    env.reset()
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(20):
        got = env._opponent_actions()
        for i, (_, agent) in enumerate(env.opponents):
            obs = agent.observe(env.games[i], 1 - env.sides[i])
            policy = agent.model.policy
            with th.no_grad():
                feats = policy.extract_features({k: th.as_tensor(v[None]) for k, v in obs.items()})
                logits = policy.action_net(policy.mlp_extractor.forward_actor(feats))[0]
            legal = obs["action_mask"].astype(bool)
            assert legal[got[i]]
            assert logits[got[i]] >= logits[th.as_tensor(legal)].max() - 1e-4
            checked += 1
        masks = np.stack(env.env_method("action_masks"))
        env.step(np.array([rng.choice(np.flatnonzero(m)) for m in masks]))
    assert env._stacked is not None and checked == 240


def test_fit_elo_orders_players_and_anchors():
    from puyotrain.rl.league import expected_score, fit_elo

    results = {
        "greedy|a": {"wins": 10, "losses": 40, "draws": 0, "timeouts": 0},
        "greedy|b": {"wins": 2, "losses": 48, "draws": 0, "timeouts": 0},
        "a|b": {"wins": 10, "losses": 40, "draws": 0, "timeouts": 0},
    }
    r = fit_elo(["greedy", "a", "b"], results, anchor="greedy")
    assert r["greedy"] == pytest.approx(1000)
    assert r["greedy"] < r["a"] < r["b"]
    # 推定した差から予想される勝率は、実際の勝率(仮想の引き分けで少し 0.5 寄り)に近い
    assert expected_score(r["a"], r["b"]) == pytest.approx(10.5 / 51, abs=0.08)
    # 全勝でも有限
    r2 = fit_elo(["x", "y"], {"x|y": {"wins": 50, "losses": 0, "draws": 0, "timeouts": 0}})
    assert np.isfinite(list(r2.values())).all()


def test_league_cli_round_robin(tmp_path, solo_ckpt):
    from puyotrain.rl.league import main

    out = tmp_path / "league.json"
    main([str(solo_ckpt.parent.parent), "--every", "128", "--include", "greedy", "random",
          "--seeds", "1", "--max-turns", "40", "--out", str(out)])
    data = json.loads(out.read_text())
    assert set(data["players"]) == {"greedy", "random", "0.000128M"}
    assert len(data["results"]) == 3
    # 2回目は新しいペアが無いので打たない
    main([str(solo_ckpt.parent.parent), "--every", "128", "--include", "greedy", "random",
          "--seeds", "1", "--max-turns", "40", "--out", str(out)])
    assert json.loads(out.read_text())["results"] == data["results"]


def test_profiler_reports_sections(tmp_path, solo_ckpt, capsys):
    from puyotrain.rl.train_versus import train

    train(versus_cfg(tmp_path, solo_ckpt, "video.enabled=false", "profile.every_iterations=1"),
          tmp_path / "run")
    out = capsys.readouterr().out
    assert "[profile] n_envs=2" in out
    assert "parameters: features_extractor=" in out
    lines = [x for x in out.splitlines() if x.startswith("[profile] 1 iters")]
    assert lines
    for name in ("env", "opp_infer", "infer", "transfer", "update", "checkpoint", "other"):
        assert f" {name}=" in lines[0]
    assert "steps/s=" in lines[0] and "samples/s=" in lines[0]


def test_timed_versus_binding():
    from puyotrain import TimedVersus

    g = TimedVersus(1)
    assert g.needs_action() == (True, True)
    assert g.timing()["chigiri_fall"][0] == 10
    assert TimedVersus(1, {"vanish": 40}).timing()["vanish"] == 40
    with pytest.raises(ValueError):
        TimedVersus(1, {"nope": 1})
    obs = versus_observation(g, 0, safe_mask=True, versus_features=True)
    assert obs["action_features"].shape == (22, _puyo.NUM_VERSUS_ACTION_FEATURES)
    assert obs["action_mask"].sum() > 0
    g.act(0, 0)  # 列0に縦置き(空の列: 12 + 2 × 11 + 20 = 54F)
    assert g.needs_action() == (False, True)
    assert g.legal_actions(0) == 0  # 操作待ちでなければ合法手なし
    g.act(1, 5)
    g.advance()
    assert g.now == 54 and g.needs_action() == (True, True)
    assert g.visible_pairs(0) == g.visible_pairs(1)  # 同じ位置なら同じツモ
    g.act(0, 0)
    g.advance()
    assert g.now == 54  # P1 が操作待ちのまま
    assert g.moves() == (1, 1)
    assert g.phases()[0] == "placing"
    # ツモの位置は着地したときに進む(P0 はまだ設置中なので P1 と同じ位置)
    assert g.visible_pairs(0) == g.visible_pairs(1)


def test_timed_match_greedy_beats_random():
    from puyotrain.rl.agents import RandomAgent
    from puyotrain.rl.match import play_match

    r = play_match(GreedyAgent(0), RandomAgent(1), [0, 1, 2], 0, timed=True)
    assert r["games"] == 6 and r["wins"] >= 5
    assert r["mean_seconds"] > 0
    assert all(g["frames"] > 0 for g in r["per_game"])


def test_greedy_pool_member_resets_per_game():
    pool = OpponentPool({"greedy": 1}, None)
    cat, agent = pool.sample()
    assert cat == "greedy" and isinstance(agent, GreedyAgent)
