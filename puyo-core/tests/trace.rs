//! トレース付き step のテスト。

use puyo_core::phase::{place_and_resolve_with, PlayerState};
use puyo_core::{
    Action, ActionMask, Field, FrameKind, Game, Pair, Puyo, Rng, Rotation, TraceFrame, Versus,
};

fn random_legal(mask: ActionMask, rng: &mut Rng) -> Action {
    let mut k = rng.below(mask.count_ones());
    let mut m = mask;
    while k > 0 {
        m &= m - 1;
        k -= 1;
    }
    Action::new(m.trailing_zeros() as usize).unwrap()
}

#[test]
fn no_chain_single_frame() {
    let mut st = PlayerState::new(0);
    let mut frames: Vec<TraceFrame> = Vec::new();
    let a = Action::from_placement(4, Rotation::Up).unwrap();
    let p = place_and_resolve_with(&mut st, Pair::new(Puyo::Red, Puyo::Blue), a, &mut frames);
    assert_eq!(p.chain.chains, 0);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].kind, FrameKind::Placed);
    assert_eq!(frames[0].field, st.field);
}

#[test]
fn chain_frame_sequence_and_scores() {
    // 列0 = R R R、列1 = B B B。列0 に (軸R, 子B) を縦置きすると、
    // R 4個が消え、子B が (0,0) に落ちて B 4個 → 2連鎖 (40 + 320)。
    let mut st = PlayerState::new(0);
    st.field = Field::from_rows(&["RB....", "RB....", "RB...."]).unwrap();
    let pair = Pair::new(Puyo::Red, Puyo::Blue);
    let mut frames: Vec<TraceFrame> = Vec::new();
    let a = Action::from_placement(0, Rotation::Up).unwrap();
    let p = place_and_resolve_with(&mut st, pair, a, &mut frames);
    assert_eq!(p.chain.chains, 2);
    assert_eq!(p.chain.score, 40 + 320);
    let kinds: Vec<_> = frames.iter().map(|f| f.kind).collect();
    use FrameKind::*;
    assert_eq!(
        kinds,
        vec![
            Placed,
            BeforeClear,
            AfterClear,
            AfterGravity,
            BeforeClear,
            AfterClear,
            AfterGravity
        ]
    );
    // 得点・連鎖数の推移
    let scores: Vec<u64> = frames.iter().map(|f| f.score).collect();
    assert_eq!(scores, vec![0, 0, 40, 40, 40, 360, 360]);
    let chains: Vec<u32> = frames.iter().map(|f| f.chain).collect();
    assert_eq!(chains, vec![0, 1, 1, 1, 2, 2, 2]);
    for w in frames.windows(2) {
        if w[0].kind == BeforeClear {
            // 消去前 → 消去後でマスクの分だけ減る
            assert_eq!(
                w[0].field.count() - w[0].cleared.count_ones(),
                w[1].field.count()
            );
            assert_eq!(w[0].cleared.count_ones(), 4);
        }
    }
    for f in &frames {
        match f.kind {
            AfterClear => {}
            _ => assert!(!f.field.has_floating(), "{:?}", f.kind),
        }
        if f.kind != BeforeClear {
            assert_eq!(f.cleared, 0);
        }
    }
    assert!(frames[2].field.has_floating());
    assert_eq!(frames.last().unwrap().field, st.field);
}

#[test]
fn traced_step_matches_step_solo() {
    for seed in 0..300u64 {
        let mut a = Game::new(seed);
        let mut b = Game::new(seed);
        let mut rng = Rng::new(seed);
        while !a.is_over() {
            let act = random_legal(a.legal_actions(), &mut rng);
            let ra = a.step(act).unwrap();
            let (rb, frames) = b.step_traced(act).unwrap();
            assert_eq!(ra, rb);
            assert_eq!(a, b);
            assert_eq!(frames[0].kind, FrameKind::Placed);
            assert_eq!(frames.len(), 1 + 3 * ra.placement.chain.chains as usize);
            assert_eq!(frames.last().unwrap().field, *a.field());
            assert_eq!(frames.last().unwrap().score, a.score());
        }
    }
}

#[test]
fn traced_step_matches_step_versus() {
    let mut drops = 0;
    for seed in 0..300u64 {
        let mut a = Versus::new(seed);
        let mut b = Versus::new(seed);
        let mut rng = Rng::new(seed);
        while !a.is_over() {
            let a0 = random_legal(a.legal_actions(0), &mut rng);
            let a1 = random_legal(a.legal_actions(1), &mut rng);
            let ra = a.step(a0, a1).unwrap();
            let (rb, frames) = b.step_traced(a0, a1).unwrap();
            assert_eq!(ra, rb);
            assert_eq!(a, b);
            for (i, f) in frames.iter().enumerate() {
                let dropped = ra.players[i].ojama_dropped > 0;
                drops += dropped as u32;
                let expected =
                    1 + 3 * ra.players[i].placement.chain.chains as usize + dropped as usize;
                assert_eq!(f.len(), expected);
                if dropped {
                    assert_eq!(f.last().unwrap().kind, FrameKind::OjamaDropped);
                }
                assert_eq!(f.last().unwrap().field, a.player(i).field);
            }
        }
    }
    assert!(drops > 0);
}
