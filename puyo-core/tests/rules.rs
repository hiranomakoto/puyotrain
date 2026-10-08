//! 得点計算・おじゃま・全消し・対戦進行のテスト。

use puyo_core::phase::{exchange, fall_ojama, place_and_resolve, PlayerState};
use puyo_core::rules::{chain_bonus, color_bonus, link_bonus};
use puyo_core::score::{attack_ojama, offset, score_to_ojama};
use puyo_core::{
    resolve_chain, Action, ChainResolver, Field, Outcome, Pair, Puyo, Rotation, Versus,
};

fn field(rows: &[&str]) -> Field {
    Field::from_rows(rows).unwrap()
}

#[test]
fn bonus_tables() {
    let chains: Vec<u32> = (1..=10).map(chain_bonus).collect();
    assert_eq!(chains, vec![0, 8, 16, 32, 64, 96, 128, 160, 192, 224]);
    let links: Vec<u32> = (4..=13).map(link_bonus).collect();
    assert_eq!(links, vec![0, 2, 3, 4, 5, 6, 7, 10, 10, 10]);
    let colors: Vec<u32> = (1..=4).map(color_bonus).collect();
    assert_eq!(colors, vec![0, 3, 6, 12]);
}

#[test]
fn four_clear_is_40() {
    let mut f = field(&["RRRR.."]);
    let s = resolve_chain(&mut f);
    assert_eq!(s.chains, 1);
    assert_eq!(s.score, 40);
    assert!(f.is_empty());
}

#[test]
fn two_chain_is_360() {
    let mut f = field(&["B.....", "R.....", "RB....", "RB....", "RB...."]);
    let steps: Vec<_> = ChainResolver::new(&mut f).collect();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].score, 40);
    assert_eq!(steps[1].score, 320);
    assert!(f.is_empty());
    let mut f2 = field(&["B.....", "R.....", "RB....", "RB....", "RB...."]);
    assert_eq!(resolve_chain(&mut f2).score, 360);
}

#[test]
fn link_bonus_score() {
    // 5連結: 5 * 10 * 2 = 100
    let mut f = field(&["RRRRR."]);
    assert_eq!(resolve_chain(&mut f).score, 100);
    // 11連結: 11 * 10 * 10 = 1100
    let mut f = field(&["RRRRR.", "RRRRRR"]);
    assert_eq!(resolve_chain(&mut f).score, 1100);
}

#[test]
fn color_and_multi_group_bonus() {
    // 2色同時(4+4): 8 * 10 * 3 = 240
    let mut f = field(&["GGGG..", "RRRR.."]);
    assert_eq!(resolve_chain(&mut f).score, 240);
    // 同色2グループ(4+4): 8 * 10 * 1 = 80(色数1 → ボーナス0、最小1)
    let mut f = field(&["RBR...", "RGR...", "RBR...", "RGR..."]);
    let s = resolve_chain(&mut f);
    assert_eq!(s.chains, 1);
    assert_eq!(s.cleared, 8);
    assert_eq!(s.score, 80);
}

#[test]
fn ojama_not_counted_in_score() {
    let mut f = field(&["OOOO..", "RRRR.."]);
    let s = resolve_chain(&mut f);
    assert_eq!(s.score, 40);
    assert_eq!(s.ojama_cleared, 4);
    assert!(f.is_empty());
}

#[test]
fn ojama_conversion_with_carry() {
    let mut carry = 0;
    assert_eq!(score_to_ojama(140, &mut carry), 2);
    assert_eq!(carry, 0);
    assert_eq!(score_to_ojama(100, &mut carry), 1);
    assert_eq!(carry, 30);
    assert_eq!(score_to_ojama(40, &mut carry), 1);
    assert_eq!(carry, 0);
    assert_eq!(score_to_ojama(69, &mut carry), 0);
    assert_eq!(score_to_ojama(1, &mut carry), 1);
    // 2連鎖 360点 → 5個、端数10
    let mut carry = 0;
    assert_eq!(score_to_ojama(360, &mut carry), 5);
    assert_eq!(carry, 10);
}

#[test]
fn offset_and_exchange() {
    let mut pending = 10;
    assert_eq!(offset(4, &mut pending), 0);
    assert_eq!(pending, 6);
    assert_eq!(offset(10, &mut pending), 4);
    assert_eq!(pending, 0);

    let mut ps = [PlayerState::new(1), PlayerState::new(2)];
    ps[0].pending = 5;
    // P0 が 12 送る → 自分の予告5を相殺、残り7。P1 は 3 送る → 打ち消して P1 に 4。
    let sent = exchange(&mut ps, [12, 3]);
    assert_eq!(sent, [4, 0]);
    assert_eq!(ps[0].pending, 0);
    assert_eq!(ps[1].pending, 4);
}

#[test]
fn ojama_drop_capped_at_30() {
    let mut st = PlayerState::new(7);
    st.pending = 45;
    assert_eq!(fall_ojama(&mut st), 30);
    assert_eq!(st.pending, 15);
    assert_eq!(st.field.count(), 30);
    assert_eq!(st.field.heights(), [5; 6]);
    assert_eq!(fall_ojama(&mut st), 15);
    assert_eq!(st.pending, 0);
    assert_eq!(st.field.count(), 45);
    assert!(!st.field.has_floating());
}

#[test]
fn ojama_overflow_above_row13_vanishes() {
    let mut st = PlayerState::new(7);
    st.pending = 90;
    for _ in 0..3 {
        fall_ojama(&mut st);
    }
    assert_eq!(st.field.heights(), [13; 6]);
    assert_eq!(st.field.count(), 78);
    assert_eq!(st.pending, 0);
}

#[test]
fn all_clear_bonus() {
    let mut carry = 0;
    assert_eq!(attack_ojama(40, &mut carry, true), 30);

    let mut st = PlayerState::new(1);
    st.field = field(&["RR...."]);
    let red = Pair::new(Puyo::Red, Puyo::Red);
    let p = place_and_resolve(
        &mut st,
        red,
        Action::from_placement(2, Rotation::Right).unwrap(),
    );
    assert!(p.all_clear);
    assert!(st.all_clear);
    assert_eq!(p.attack, 0); // 40点 → 0個(端数40)

    // 連鎖なしの設置ではボーナスは消費されない。
    let green = Pair::new(Puyo::Green, Puyo::Blue);
    let p = place_and_resolve(
        &mut st,
        green,
        Action::from_placement(0, Rotation::Up).unwrap(),
    );
    assert_eq!(p.attack, 0);
    assert!(st.all_clear);

    // 次の連鎖で 30個上乗せ: 40 + 端数40 = 80点 → 1個 + 30個。
    st.field = field(&["RR....", "GB...."]);
    let p = place_and_resolve(
        &mut st,
        red,
        Action::from_placement(2, Rotation::Up).unwrap(),
    );
    assert_eq!(p.chain.chains, 1);
    assert_eq!(p.attack, 31);
    assert!(!st.all_clear);
}

/// 列0に縦置きすると、現在の組ぷよの軸色がちょうど4個つながって1連鎖(40点)になる盤面。
fn chain_ready(v: &Versus) -> Field {
    let pair = v.pair(0);
    let n = if pair.axis == pair.child { 2 } else { 3 };
    let mut f = Field::new();
    for _ in 0..n {
        f.push(0, pair.axis);
    }
    f
}

#[test]
fn ojama_falls_after_one_turn_grace() {
    let mut v = Versus::new(123);
    v.player_mut(0).field = chain_ready(&v);
    v.player_mut(0).carry = 700 - 40; // 40点 + 660 = 700点 → 10個
    let up0 = Action::from_placement(0, Rotation::Up).unwrap();
    let up5 = Action::from_placement(5, Rotation::Up).unwrap();

    let r = v.step(up0, up5).unwrap();
    assert_eq!(r.players[0].placement.attack, 10);
    assert_eq!(r.players[0].ojama_sent, 10);
    // このターンにはまだ降らない(予告のみ)
    assert_eq!(r.players[1].ojama_dropped, 0);
    assert_eq!(v.player(1).pending, 10);
    assert_eq!(v.player(1).field.count(), 2);

    // 次のターン、連鎖しなければ降る
    let r = v.step(up5, up5).unwrap();
    assert_eq!(r.players[1].ojama_dropped, 10);
    assert_eq!(v.player(1).pending, 0);
    assert_eq!(v.player(1).field.count(), 4 + 10);
}

#[test]
fn ojama_can_be_offset_during_grace() {
    let mut v = Versus::new(5);
    v.player_mut(0).field = chain_ready(&v);
    v.player_mut(0).carry = 700 - 40; // 10個
    let up0 = Action::from_placement(0, Rotation::Up).unwrap();
    let up5 = Action::from_placement(5, Rotation::Up).unwrap();
    v.step(up0, up5).unwrap();
    assert_eq!(v.player(1).pending, 10);

    // P1 が返す: 列0に縦置きで連鎖、得点 40 + 端数 → 14個
    v.player_mut(1).field = chain_ready(&v);
    v.player_mut(1).carry = 14 * 70 - 40;
    let r = v.step(up5, up0).unwrap();
    assert_eq!(r.players[1].placement.attack, 14);
    // 連鎖したので降らない。10個相殺して4個を送り返す。
    assert_eq!(r.players[1].ojama_dropped, 0);
    assert_eq!(v.player(1).pending, 0);
    assert_eq!(r.players[1].ojama_sent, 4);
    assert_eq!(v.player(0).pending, 4);
}

#[test]
fn simultaneous_attacks_cancel() {
    let mut v = Versus::new(9);
    let f = chain_ready(&v);
    v.player_mut(0).field = f;
    v.player_mut(1).field = f;
    v.player_mut(0).carry = 700 - 40; // 10個
    v.player_mut(1).carry = 420 - 40; // 6個
    let up0 = Action::from_placement(0, Rotation::Up).unwrap();
    let r = v.step(up0, up0).unwrap();
    assert_eq!(r.players[0].ojama_sent, 4);
    assert_eq!(r.players[1].ojama_sent, 0);
    assert_eq!(v.player(1).pending, 4);
    assert_eq!(v.player(0).pending, 0);
}

#[test]
fn death_and_draw() {
    let mut v = Versus::new(1);
    // 3列目を高さ10にしておくと、縦置きで12段目が埋まり窒息する。
    let mut f = Field::new();
    let fill = [
        Puyo::Red,
        Puyo::Green,
        Puyo::Blue,
        Puyo::Yellow,
        Puyo::Ojama,
    ];
    for i in 0..10 {
        f.push(2, fill[i % 5]);
    }
    // 連鎖が起きないよう色をずらして確認
    assert!(resolve_chain(&mut f.clone()).chains == 0);
    v.player_mut(0).field = f;
    v.player_mut(1).field = f;
    let up2 = Action::from_placement(2, Rotation::Up).unwrap();
    let up5 = Action::from_placement(5, Rotation::Up).unwrap();

    let mut a = v;
    let r = a.step(up2, up5).unwrap();
    assert_eq!(r.outcome, Some(Outcome::Win(1)));
    assert!(a.step(up5, up5).is_err());
    assert_eq!(a.legal_actions(0), 0);

    let mut b = v;
    let r = b.step(up2, up2).unwrap();
    assert_eq!(r.outcome, Some(Outcome::Draw));
}

#[test]
fn illegal_action_rejected() {
    let mut v = Versus::new(1);
    let mut f = Field::new();
    for i in 0..12 {
        f.push(1, if i % 2 == 0 { Puyo::Red } else { Puyo::Green });
    }
    v.player_mut(1).field = f;
    let left = Action::from_placement(0, Rotation::Up).unwrap();
    let ok = Action::from_placement(3, Rotation::Up).unwrap();
    assert!(v.legal_actions(1) & (1 << left.index()) == 0);
    let before = v;
    assert!(v.step(ok, left).is_err());
    assert_eq!(v, before, "illegal action must not change state");
}
