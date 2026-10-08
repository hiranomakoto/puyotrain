import copy

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import puyotrain
from puyotrain import Game, PuyoSoloEnv, PuyoVersusEnv, Versus


def random_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(mask)
    assert legal.size > 0
    return int(rng.choice(legal))


# ---------------------------------------------------------------- core bindings


def test_game_basic():
    g = Game(1)
    assert g.legal_actions() == (1 << 22) - 1
    assert g.action_mask().tolist() == [1] * 22
    score, chains, all_clear, dead = g.step(0)
    assert (score, chains, all_clear, dead) == (0, 0, False, False)
    assert g.turn == 1
    assert sum(c not in ".\n" for c in g.field_str()) == 2


def test_illegal_and_out_of_range_actions():
    g = Game(1)
    with pytest.raises(ValueError):
        g.step(22)
    # 列2(0始まり)の左側を高くして列0を塞ぐ
    while g.legal_actions() & 1:
        g.step(1)  # 2列目に縦置きを続ける
        if g.is_over:
            pytest.skip("died before blocking")
    with pytest.raises(ValueError):
        g.step(0)


def test_game_over_raises():
    g = Game(3)
    while not g.is_over:
        g.step(2)  # 3列目に積み続けると窒息
    with pytest.raises(RuntimeError):
        g.step(2)
    assert g.legal_actions() == 0


def test_clone_independent():
    g = Game(5)
    g.step(3)
    c = g.clone()
    c.step(4)
    assert g.turn == 1 and c.turn == 2
    d = copy.deepcopy(g)
    assert d.field_str() == g.field_str()

    v = Versus(5)
    w = v.clone()
    w.step(0, 1)
    assert v.turn == 0 and w.turn == 1


def test_same_tsumo_for_both_players():
    v = Versus(11)
    g = Game(11)
    assert v.visible_pairs() == g.visible_pairs()


# ---------------------------------------------------------------- observations


def test_observation_layout():
    g = Game(7)
    pairs = g.visible_pairs()
    obs = g.observation()
    assert obs["board"].shape == (2, 6, 13, 6)
    assert obs["pieces"].shape == (3, 2, 4)
    # 自分の盤面は全セルちょうど1平面
    assert (obs["board"][0].sum(axis=0) == 1).all()
    # 1人用では相手盤面はすべて0
    assert obs["board"][1].sum() == 0
    for k, (axis, child) in enumerate(pairs):
        assert obs["pieces"][k, 0].tolist() == [int(i == axis) for i in range(4)]
        assert obs["pieces"][k, 1].tolist() == [int(i == child) for i in range(4)]

    # 縦置き(子が上)で3列目へ: 軸が1段目、子が2段目
    axis, child = pairs[0]
    g.step(2)
    b = g.observation()["board"][0]
    assert b[1 + axis, 0, 2] == 1
    assert b[1 + child, 1, 2] == 1
    assert b[0, 2, 2] == 1  # 3段目は空
    assert b[1:, :, :].sum() == 2


def test_versus_observation_perspective():
    v = Versus(3)
    v.step(0, 5)
    o0, o1 = v.observation(0), v.observation(1)
    assert (o0["board"][0] == o1["board"][1]).all()
    assert (o0["board"][1] == o1["board"][0]).all()
    assert o0["board"][0, 1:, :, 0].sum() == 2  # P0 は列0
    assert o1["board"][0, 1:, :, 5].sum() == 2  # P1 は列5


# ---------------------------------------------------------------- solo env


def test_solo_check_env():
    check_env(PuyoSoloEnv(), skip_render_check=True)


def test_solo_registered():
    env = gym.make("puyotrain/PuyoSolo-v0")
    obs, _ = env.reset(seed=0)
    assert "board" in obs


def test_solo_reward_is_score():
    env = PuyoSoloEnv()
    rng = np.random.default_rng(0)
    for ep in range(30):
        obs, _ = env.reset(seed=ep)
        total = 0.0
        while True:
            obs, r, term, trunc, info = env.step(random_action(obs["action_mask"], rng))
            assert r >= 0 and r % 10 == 0
            total += r
            assert info["total_score"] == total
            assert env.observation_space.contains(obs)
            if term or trunc:
                break
        assert term


def test_solo_reproducible():
    def run(seed):
        env = PuyoSoloEnv()
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        trace = []
        done = False
        while not done:
            a = random_action(obs["action_mask"], rng)
            obs, r, term, trunc, _ = env.step(a)
            trace.append((a, r, obs["board"].tobytes()))
            done = term or trunc
        return trace

    assert run(42) == run(42)
    assert run(42) != run(43)


def test_solo_truncation():
    env = PuyoSoloEnv(max_steps=3)
    obs, _ = env.reset(seed=0)
    for i in range(3):
        obs, _, term, trunc, _ = env.step(int(np.flatnonzero(obs["action_mask"])[-1]))
    assert trunc and not term


def test_solo_render():
    env = PuyoSoloEnv(render_mode="ansi")
    env.reset(seed=0)
    out = env.render()
    assert out.count("\n") == 13


# ---------------------------------------------------------------- versus env


def play_versus(env: PuyoVersusEnv, seed: int, agent_seed: int):
    (o0, o1), _ = env.reset(seed=seed)
    rng = np.random.default_rng(agent_seed)
    trace = []
    while True:
        a0 = random_action(o0["action_mask"], rng)
        a1 = random_action(o1["action_mask"], rng)
        (o0, o1), (r0, r1), term, trunc, info = env.step(a0, a1)
        trace.append((a0, a1, r0, r1, o0["board"].tobytes(), tuple(o0["ojama"])))
        if term or trunc:
            return trace, (r0, r1), term, trunc, info


def test_versus_reproducible():
    env = PuyoVersusEnv()
    t1 = play_versus(env, 123, 9)
    t2 = play_versus(env, 123, 9)
    assert t1[0] == t2[0]
    assert t1[1] == t2[1]


def test_versus_ojama_observed():
    env = PuyoVersusEnv()
    seen = False
    for seed in range(50):
        trace, *_ = play_versus(env, seed, seed)
        if any(o != (0, 0) for *_, o in trace):
            seen = True
            break
    assert seen, "random play should send ojama at least once in 50 games"


def test_versus_truncation():
    env = PuyoVersusEnv(max_turns=2)
    _, rewards, term, trunc, _ = play_versus(env, 0, 0)
    assert trunc and not term and rewards == (0.0, 0.0)


def test_versus_1000_random_games():
    env = PuyoVersusEnv()
    rng = np.random.default_rng(2024)
    counts = {"p0": 0, "p1": 0, "draw": 0}
    for g in range(1000):
        (o0, o1), _ = env.reset(seed=g)
        while True:
            a0 = random_action(o0["action_mask"], rng)
            a1 = random_action(o1["action_mask"], rng)
            (o0, o1), (r0, r1), term, trunc, info = env.step(a0, a1)
            assert not trunc
            if term:
                break
            assert (r0, r1) == (0.0, 0.0)
        assert r0 == -r1
        if info["draw"]:
            assert (r0, r1) == (0.0, 0.0)
            counts["draw"] += 1
        else:
            assert {r0, r1} == {1.0, -1.0}
            counts["p0" if info["winner"] == 0 else "p1"] += 1
    assert sum(counts.values()) == 1000


def test_versus_spaces_contain_obs():
    env = PuyoVersusEnv()
    (o0, o1), _ = env.reset(seed=1)
    assert env.observation_space.contains(o0)
    assert env.observation_space.contains(o1)
    m0, m1 = env.action_masks()
    assert m0.dtype == bool and m0.all()


def test_module_constants():
    assert puyotrain.NUM_ACTIONS == 22
    assert (puyotrain.WIDTH, puyotrain.HEIGHT) == (6, 13)
