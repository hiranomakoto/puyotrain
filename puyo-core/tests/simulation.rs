//! プロパティテスト、再現性、ランダム対戦の完走。

use proptest::prelude::*;
use puyo_core::{Action, ActionMask, ChainResolver, Field, Game, Puyo, Rng, Versus};

fn random_legal(mask: ActionMask, rng: &mut Rng) -> Action {
    assert!(mask != 0, "no legal action while game is running");
    let n = mask.count_ones();
    let mut k = rng.below(n);
    let mut m = mask;
    loop {
        let i = m.trailing_zeros();
        if k == 0 {
            return Action::new(i as usize).unwrap();
        }
        k -= 1;
        m &= m - 1;
    }
}

fn arb_field() -> impl Strategy<Value = Field> {
    // 各セルを空 or 5種のいずれかにし、重力で落とす(13段目まで使う)。
    prop::collection::vec(0u8..6, 6 * 13).prop_map(|cells| {
        let mut f = Field::new();
        for (i, c) in cells.into_iter().enumerate() {
            if let Some(p) = Puyo::from_index(c as usize) {
                f.set(i % 6, i / 6, Some(p));
            }
        }
        f.apply_gravity();
        f
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn no_floating_after_each_chain_step(mut f in arb_field()) {
        prop_assert!(!f.has_floating());
        let mut steps = 0;
        let mut it = ChainResolver::new(&mut f);
        while it.next().is_some() {
            steps += 1;
            prop_assert!(steps < 40);
        }
        prop_assert!(!f.has_floating());
        // 連鎖後に消せるグループは残っていない
        let mut g = f;
        prop_assert!(g.clear_groups().is_none());
    }

    #[test]
    fn no_floating_after_random_play(seed in any::<u64>(), choices in prop::collection::vec(any::<u64>(), 1..200)) {
        let mut g = Game::new(seed);
        let mut rng = Rng::new(seed ^ 0xabcdef);
        for _ in choices {
            if g.is_over() { break; }
            let a = random_legal(g.legal_actions(), &mut rng);
            g.step(a).unwrap();
            prop_assert!(!g.field().has_floating());
        }
    }

    #[test]
    fn versus_no_floating(seed in any::<u64>()) {
        let mut v = Versus::new(seed);
        let mut rng = Rng::new(seed);
        while !v.is_over() {
            let a0 = random_legal(v.legal_actions(0), &mut rng);
            let a1 = random_legal(v.legal_actions(1), &mut rng);
            v.step(a0, a1).unwrap();
            prop_assert!(!v.player(0).field.has_floating());
            prop_assert!(!v.player(1).field.has_floating());
        }
    }
}

fn play_versus(seed: u64, agent_seed: u64) -> (Versus, Vec<(Action, Action)>) {
    let mut v = Versus::new(seed);
    let mut rng = Rng::new(agent_seed);
    let mut log = Vec::new();
    while !v.is_over() {
        let a0 = random_legal(v.legal_actions(0), &mut rng);
        let a1 = random_legal(v.legal_actions(1), &mut rng);
        v.step(a0, a1).unwrap();
        log.push((a0, a1));
        assert!(v.turn() < 10_000);
    }
    (v, log)
}

#[test]
fn same_seed_reproduces() {
    let (a, log_a) = play_versus(2024, 7);
    let (b, log_b) = play_versus(2024, 7);
    assert_eq!(a, b);
    assert_eq!(log_a, log_b);

    // 行動列を再生しても同じ結果になる
    let mut c = Versus::new(2024);
    for (a0, a1) in &log_a {
        c.step(*a0, *a1).unwrap();
    }
    assert_eq!(a, c);

    // ツモは両者共通かつシードで決まる
    let g1 = Game::new(99);
    let g2 = Game::new(99);
    let seq1: Vec<_> = (0..100).map(|k| g1.pair(k)).collect();
    let seq2: Vec<_> = (0..100).map(|k| g2.pair(k)).collect();
    assert_eq!(seq1, seq2);
    let g3 = Game::new(100);
    let seq3: Vec<_> = (0..100).map(|k| g3.pair(k)).collect();
    assert_ne!(seq1, seq3);
}

#[test]
fn clone_is_independent() {
    let (mid, _) = {
        let mut v = Versus::new(3);
        let mut rng = Rng::new(3);
        for _ in 0..10 {
            let a0 = random_legal(v.legal_actions(0), &mut rng);
            let a1 = random_legal(v.legal_actions(1), &mut rng);
            v.step(a0, a1).unwrap();
        }
        (v, ())
    };
    let snapshot = mid;
    let mut branch = mid;
    let a = Action::new(0).unwrap();
    if branch.legal_actions(0) & 1 != 0 && branch.legal_actions(1) & 1 != 0 {
        branch.step(a, a).unwrap();
        assert_ne!(branch, snapshot);
    }
    assert_eq!(mid, snapshot);
}

#[test]
fn random_agents_1000_games() {
    let mut wins = [0u32; 2];
    let mut draws = 0;
    let mut total_turns = 0;
    let mut max_chain = 0;
    for g in 0..1000u64 {
        let mut v = Versus::new(g);
        let mut rng = Rng::new(g.wrapping_mul(31) + 1);
        while !v.is_over() {
            let a0 = random_legal(v.legal_actions(0), &mut rng);
            let a1 = random_legal(v.legal_actions(1), &mut rng);
            let r = v.step(a0, a1).unwrap();
            for p in &r.players {
                max_chain = max_chain.max(p.placement.chain.chains);
            }
            assert!(v.turn() < 10_000);
        }
        total_turns += v.turn();
        match v.outcome().unwrap() {
            puyo_core::Outcome::Win(i) => wins[i] += 1,
            puyo_core::Outcome::Draw => draws += 1,
        }
    }
    assert_eq!(wins[0] + wins[1] + draws, 1000);
    eprintln!(
        "1000 games: wins={:?} draws={} avg_turns={:.1} max_chain={}",
        wins,
        draws,
        total_turns as f64 / 1000.0,
        max_chain
    );
}

#[test]
fn random_solo_games() {
    for s in 0..200u64 {
        let mut g = Game::new(s);
        let mut rng = Rng::new(s + 1);
        while !g.is_over() {
            let a = random_legal(g.legal_actions(), &mut rng);
            g.step(a).unwrap();
        }
        assert!(g.step(Action::new(0).unwrap()).is_err());
    }
}
