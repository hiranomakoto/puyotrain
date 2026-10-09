"""学習基盤(トレース、ベースライン、リプレイ、動画、学習)のテスト。

RL 用の依存(pyyaml, pillow, imageio, torch, sb3-contrib)が無い環境ではスキップする。
"""

import json

import numpy as np
import pytest

from puyotrain import Game, PuyoSoloEnv, Versus

yaml = pytest.importorskip("yaml")
pytest.importorskip("PIL")
pytest.importorskip("imageio_ffmpeg")

from puyotrain.rl.agents import GreedyAgent, RandomAgent, make_agent  # noqa: E402
from puyotrain.rl.config import load_config  # noqa: E402
from puyotrain.rl.evaluate import evaluate, play_solo  # noqa: E402
from puyotrain.rl.render import render_replay, solo_scenes, versus_scenes  # noqa: E402
from puyotrain.rl.replay import load_replay, record_solo, record_versus, save_replay  # noqa: E402

PUYO = {"R": 0, "G": 1, "B": 2, "Y": 3, "O": 4}


def board_from_str(s: str) -> np.ndarray:
    rows = s.strip().splitlines()[::-1]
    return np.array([[PUYO.get(c, -1) for c in row] for row in rows], dtype=np.int8)


def small_video_cfg() -> dict:
    cfg = load_config()["video"]
    cfg.update(frames_per_move=1, frames_per_chain_stage=1, end_hold_frames=1)
    return cfg


# ---------------------------------------------------------------- trace


def test_step_traced_matches_step():
    rng = np.random.default_rng(0)
    for seed in range(30):
        a, b = Game(seed), Game(seed)
        while not a.is_over:
            legal = [i for i in range(22) if a.legal_actions() >> i & 1]
            act = int(rng.choice(legal))
            ra = a.step(act)
            rb, frames = b.step_traced(act)
            assert ra == rb
            kinds = [f["kind"] for f in frames]
            assert kinds[0] == "placed"
            assert kinds[1:] == ["before_clear", "after_clear", "after_gravity"] * ra[1]
            assert (frames[-1]["board"] == a.board()).all()
            assert frames[-1]["score"] == a.score
            for f in frames:
                if f["kind"] == "before_clear":
                    assert f["cleared"].sum() >= 4
                    assert (f["board"][f["cleared"]] >= 0).all()
                else:
                    assert not f["cleared"].any()


def test_board_matches_field_str():
    g = Game(4)
    for a in (0, 5, 12, 20):
        g.step(a)
    assert (g.board() == board_from_str(g.field_str())).all()


def test_versus_step_traced_has_ojama_frames():
    rng = np.random.default_rng(1)
    seen = False
    for seed in range(40):
        v = Versus(seed)
        while not v.is_over:
            acts = [int(rng.choice([i for i in range(22) if v.legal_actions(p) >> i & 1]))
                    for p in range(2)]
            res, frames = v.step_traced(*acts)
            for p in range(2):
                assert (frames[p][-1]["board"] == v.board(p)).all()
                if res[p]["ojama_dropped"]:
                    assert frames[p][-1]["kind"] == "ojama_dropped"
                    seen = True
    assert seen


# ---------------------------------------------------------------- baselines


def test_greedy_reproducible_and_beats_random():
    seeds = list(range(8))
    g1 = evaluate(GreedyAgent(), seeds, 150)
    g2 = evaluate(GreedyAgent(), seeds, 150)
    r = evaluate(RandomAgent(), seeds, 150)
    assert g1["mean_score"] == g2["mean_score"]
    assert g1["mean_score"] > 3 * r["mean_score"]
    assert g1["mean_max_chain"] >= r["mean_max_chain"]


def test_agents_reset_per_seed():
    a = RandomAgent(0)
    x = play_solo(a, 5, 30)
    play_solo(a, 6, 30)
    y = play_solo(a, 5, 30)
    assert x["actions"] == y["actions"]


# ---------------------------------------------------------------- env options


def test_env_game_seed_and_reward_scale():
    env = PuyoSoloEnv(reward_scale=0.001)
    obs, info = env.reset(options={"game_seed": 77})
    assert info["game_seed"] == 77
    assert Game(77).visible_pairs() == [tuple(p) for p in env.game.visible_pairs()]
    total = 0.0
    done = False
    rng = np.random.default_rng(0)
    while not done:
        obs, r, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        total += r
        done = term or trunc
    assert abs(total - info["total_score"] * 0.001) < 1e-9
    assert info["max_chain"] >= 0


# ---------------------------------------------------------------- replay & render


def test_solo_replay_roundtrip(tmp_path):
    rep = record_solo(make_agent("greedy"), 3, 60)
    path = save_replay(rep, tmp_path / "r.json")
    loaded = load_replay(path)
    assert loaded == json.loads(json.dumps(rep))
    g = Game(loaded["seed"])
    for a in loaded["actions"]:
        g.step(a)
    assert g.score == loaded["meta"]["final_scores"][0]
    assert g.turn == loaded["meta"]["moves"]


def test_versus_replay_roundtrip(tmp_path):
    rep = record_versus((make_agent("greedy"), make_agent("random")), 2, 300)
    loaded = load_replay(save_replay(rep, tmp_path / "v.json"))
    v = Versus(loaded["seed"])
    for a0, a1 in loaded["actions"]:
        v.step(a0, a1)
    assert list(v.scores()) == loaded["meta"]["final_scores"]
    assert loaded["meta"]["result"] in ("p1_win", "p2_win", "draw", "max_moves")


def test_scenes_follow_trace():
    cfg = small_video_cfg()
    rep = record_solo(make_agent("greedy"), 0, 40)
    scenes = list(solo_scenes(rep, cfg))
    assert scenes[-1].players[0].score == rep["meta"]["final_scores"][0]
    assert max(s.players[0].max_chain for s in scenes) == rep["meta"]["max_chains"][0]
    rep = record_versus((make_agent("greedy"), make_agent("greedy")), 0, 30)
    scenes = list(versus_scenes(rep, cfg))
    assert all(len(s.players) == 2 for s in scenes)


def test_render_mp4(tmp_path):
    cfg = small_video_cfg()
    rep = record_solo(make_agent("random"), 1, 15)
    out = render_replay(rep, tmp_path / "solo.mp4", cfg)
    assert out.exists() and out.stat().st_size > 1000
    rep = record_versus((make_agent("random"), make_agent("greedy")), 1, 15)
    out = render_replay(rep, tmp_path / "vs.mp4", cfg)
    assert out.exists() and out.stat().st_size > 1000


# ---------------------------------------------------------------- training (torch)


def test_tiny_training_run(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from puyotrain.rl.agents import PPOAgent
    from puyotrain.rl.render_all import render_all
    from puyotrain.rl.train import train

    cfg = load_config(overrides=[
        "env.n_envs=2", "env.vec_env=dummy", "ppo.total_timesteps=256", "ppo.n_steps=64",
        "ppo.batch_size=64", "ppo.n_epochs=1", "checkpoint.every_steps=128",
        "eval.seeds.count=3", "eval.max_steps=30", "eval.baselines=[random]",
        "video.seeds=[0]", "video.max_moves=10", "video.frames_per_move=1",
        "video.frames_per_chain_stage=1", "video.end_hold_frames=1",
    ])
    run = train(cfg, tmp_path / "run")
    ckpts = sorted((run / "checkpoints").glob("step_*.zip"))
    assert [c.stem for c in ckpts] == ["step_0000000128", "step_0000000256"]
    for c in ckpts:
        meta = json.loads(c.with_suffix(".json").read_text())
        assert meta["step"] == int(c.stem.split("_")[1])
        assert {"mean_score", "mean_max_chain"} <= meta["metrics"].keys()
        assert "score_vs_random" in meta["comparison"]
        assert meta["config"]["ppo"]["total_timesteps"] == 256
        assert (run / "videos" / f"{c.stem}_seed0.mp4").exists()
    assert (run / "videos" / "baseline_random_seed0.mp4").exists()
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 2

    # バッチ推論と1局ずつの推論が一致する
    agent = PPOAgent(ckpts[-1])
    a = evaluate(agent, [0, 1, 2], 30)
    games = [play_solo(agent, s, 30) for s in (0, 1, 2)]
    assert a["mean_score"] == np.mean([g["score"] for g in games])

    # 動画をまとめて作り直す
    for mp4 in (run / "videos").glob("*.mp4"):
        mp4.unlink()
    made = render_all(run)
    assert len(made) == 3  # baseline_random + 2 チェックポイント


def _tiny_overrides(total: int) -> list[str]:
    return [
        "env.n_envs=2", "env.vec_env=dummy", f"ppo.total_timesteps={total}", "ppo.n_steps=64",
        "ppo.batch_size=64", "ppo.n_epochs=1", "checkpoint.every_steps=128",
        "eval.seeds.count=2", "eval.max_steps=20", "eval.baselines=[random]",
        "video.enabled=false",
    ]


def test_resume_continues_same_run(tmp_path):
    pytest.importorskip("sb3_contrib")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from puyotrain.rl.train import resume, train

    run = train(load_config(overrides=_tiny_overrides(256)), tmp_path / "run")
    assert sorted(p.stem for p in (run / "checkpoints").glob("*.zip")) == [
        "step_0000000128", "step_0000000256"]
    first_256 = (run / "checkpoints" / "step_0000000256.zip").read_bytes()

    # 目標を上げて再開すると、同じランに続きのチェックポイントが増える(古いものはそのまま)
    resume(run, ["ppo.total_timesteps=512", "ppo.learning_rate=0.0001"])
    names = sorted(p.stem for p in (run / "checkpoints").glob("*.zip"))
    assert names == ["step_0000000128", "step_0000000256", "step_0000000384", "step_0000000512"]
    assert (run / "checkpoints" / "step_0000000256.zip").read_bytes() == first_256
    meta = json.loads((run / "checkpoints" / "step_0000000512.json").read_text())
    assert meta["resumed_from"].endswith("step_0000000256.zip")
    assert meta["config"]["ppo"]["learning_rate"] == 0.0001
    steps = [json.loads(l)["step"] for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert steps == [128, 256, 384, 512]
    assert len((run / "resume_history.jsonl").read_text().splitlines()) == 1

    # TensorBoard は同じディレクトリに続けて書かれる
    tb_dirs = list((run / "tb").iterdir())
    assert len(tb_dirs) == 1
    ea = EventAccumulator(str(tb_dirs[0]))
    ea.Reload()
    assert [e.step for e in ea.Scalars("eval/mean_score")] == [128, 256, 384, 512]

    # 目標に到達済みなら再開しない
    with pytest.raises(ValueError, match="raise it"):
        resume(run, [])
    # ネットワークの形は変えられない
    with pytest.raises(ValueError, match="policy"):
        resume(run, ["ppo.total_timesteps=1024", "policy.features_dim=64"])
    # 古いチェックポイントから同じランには続けられない(後のチェックポイントを上書きするため)
    old = run / "checkpoints" / "step_0000000128.zip"
    with pytest.raises(ValueError, match="--run-dir"):
        resume(run, ["ppo.total_timesteps=1024"], from_checkpoint=old)
    # 分岐先を指定すれば、そこから別のランとして続けられる
    branch = resume(run, ["ppo.total_timesteps=256"], from_checkpoint=old,
                    new_run_dir=tmp_path / "branch")
    assert sorted(p.stem for p in (branch / "checkpoints").glob("*.zip")) == ["step_0000000256"]
    assert (run / "checkpoints" / "step_0000000512.zip").exists()


def test_video_every_checkpoints(tmp_path):
    pytest.importorskip("sb3_contrib")
    from puyotrain.rl.train import train

    ov = _tiny_overrides(384)
    ov.remove("video.enabled=false")
    ov += ["video.every_checkpoints=2", "video.seeds=[0]", "video.max_moves=5",
           "video.frames_per_move=1", "video.frames_per_chain_stage=1", "video.end_hold_frames=1"]
    run = train(load_config(overrides=ov), tmp_path / "run")
    videos = sorted(p.name for p in (run / "videos").glob("step_*.mp4"))
    # 128 は飛ばし、256 は2回に1回、384 は学習終了時なので作る
    assert videos == ["step_0000000256_seed0.mp4", "step_0000000384_seed0.mp4"]


def test_old_run_config_gets_new_default_keys(tmp_path):
    cfg = load_config()
    del cfg["video"]["every_checkpoints"]
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    loaded = load_config(p, ["video.every_checkpoints=5"])
    assert loaded["video"]["every_checkpoints"] == 5
