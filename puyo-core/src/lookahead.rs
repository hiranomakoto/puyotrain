//! 先読みの特徴量: 手ごとに、見えている組ぷよ(現在・ネクスト・ネクネク)を3手先まで置いてみた結果。
//!
//! 練習の対局で、3手先まで全部試す探索(`builder`)は AI よりずっと大きな連鎖の種を育てた。
//! その先読みの結果を手ごとの特徴量にして AI に見せる。全部試すと1局面あたり約22ms かかるので、
//! 2手目は「2手目の後の潜在連鎖」(同じなら撃てた連鎖)が上位 [`LOOK_TOP_K`] 個の手だけを
//! 3手目まで展開する。
//!
//! 手ごとの3個(合法でない手、置いた直後に窒息する手は全部0):
//!
//! - look_potential_chains: 3手先までに育つ連鎖の種(潜在連鎖)の最大(/10)
//! - look_fire_chains: 3手先までに撃てる連鎖の最大(この手の連鎖も含む。/10)
//! - look_survivable: 3手先まで窒息せずに置ける並びがあれば1
//!   (展開しなかった2手目の先は数えないので、まれに0になりうる)

use crate::action::{legal_actions, place_pair, Action};
use crate::chain::resolve_chain;
use crate::color::Pair;
use crate::field::Field;
use crate::potential::potential_chain;
use crate::rules::NUM_ACTIONS;

/// 先読みの特徴量の数。
pub const NUM_LOOK_FEATURES: usize = 3;

/// 先読みの特徴量の名前(並び順)。
pub const LOOK_FEATURE_NAMES: [&str; NUM_LOOK_FEATURES] = [
    "look_potential_chains",
    "look_fire_chains",
    "look_survivable",
];

/// 2手目のうち、3手目まで展開する数。
pub const LOOK_TOP_K: usize = 4;

/// 置いて連鎖を解く。戻り値: (盤面, 連鎖数)。窒息なら None。
#[inline]
fn play(field: &Field, pair: Pair, action: Action) -> Option<(Field, u32)> {
    let mut f = *field;
    place_pair(&mut f, pair, action);
    let chains = resolve_chain(&mut f).chains;
    (!f.is_dead()).then_some((f, chains))
}

/// 1手目を置いた後の盤面 `f1`(1手目で撃った連鎖 `fired1`)からの先読み。
fn look_from(f1: &Field, fired1: u32, pairs: [Pair; 2]) -> [f32; NUM_LOOK_FEATURES] {
    // 2手目: 全部試して、潜在連鎖(同じなら撃てた連鎖)の順に並べる
    let mut second: Vec<(u32, u32, Field)> = Vec::with_capacity(NUM_ACTIONS);
    let mut fire = fired1;
    let mut m = legal_actions(f1);
    while m != 0 {
        let a = Action::new(m.trailing_zeros() as usize).unwrap();
        m &= m - 1;
        if let Some((f2, c2)) = play(f1, pairs[0], a) {
            fire = fire.max(c2);
            second.push((potential_chain(&f2).chains, c2, f2));
        }
    }
    second.sort_unstable_by_key(|x| std::cmp::Reverse((x.0, x.1)));
    // 3手目: 上位だけ展開する
    let mut pot = 0;
    let mut survivable = false;
    for (pot2, _, f2) in second.iter().take(LOOK_TOP_K) {
        pot = pot.max(*pot2);
        let mut m = legal_actions(f2);
        while m != 0 {
            let a = Action::new(m.trailing_zeros() as usize).unwrap();
            m &= m - 1;
            if let Some((f3, c3)) = play(f2, pairs[1], a) {
                survivable = true;
                fire = fire.max(c3);
                pot = pot.max(potential_chain(&f3).chains);
            }
        }
    }
    [
        pot as f32 / 10.0,
        fire as f32 / 10.0,
        survivable as u8 as f32,
    ]
}

/// 手ごとの先読みの特徴量。`pairs` は現在・ネクスト・ネクネク。
pub fn lookahead_features(
    field: &Field,
    pairs: [Pair; 3],
) -> [[f32; NUM_LOOK_FEATURES]; NUM_ACTIONS] {
    let mut out = [[0.0; NUM_LOOK_FEATURES]; NUM_ACTIONS];
    let mut m = legal_actions(field);
    while m != 0 {
        let a = Action::new(m.trailing_zeros() as usize).unwrap();
        m &= m - 1;
        if let Some((f1, c1)) = play(field, pairs[0], a) {
            out[a.index()] = look_from(&f1, c1, [pairs[1], pairs[2]]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Puyo;
    use crate::rules::NUM_COLORS;

    const P: [Pair; 3] = [
        Pair::new(Puyo::Red, Puyo::Green),
        Pair::new(Puyo::Blue, Puyo::Yellow),
        Pair::new(Puyo::Red, Puyo::Red),
    ];

    #[test]
    fn empty_field_is_survivable_and_illegal_is_zero() {
        let f = Field::new();
        let feats = lookahead_features(&f, P);
        let legal = legal_actions(&f);
        for (i, row) in feats.iter().enumerate() {
            if legal >> i & 1 == 1 {
                assert_eq!(row[2], 1.0);
                assert!(row[0] >= 0.0);
            } else {
                assert_eq!(*row, [0.0; NUM_LOOK_FEATURES]);
            }
        }
        assert_eq!(NUM_COLORS, 4);
    }

    #[test]
    fn sees_a_chain_fired_by_the_third_pair() {
        // 列0 に R が2個。3手目の RR を列0に縦置きすると消える
        let f = Field::from_rows(&["R.....", "R....."]).unwrap();
        let feats = lookahead_features(&f, P);
        assert!(feats.iter().any(|r| r[1] >= 0.1));
    }

    #[test]
    fn lookahead_potential_is_at_least_the_search_seed() {
        // 3手先の最大の潜在連鎖は、どの手でも「その手の直後の潜在連鎖」以上の手がある
        let f = Field::from_rows(&["G.....", "RR....", "RGGG.."]).unwrap();
        let feats = lookahead_features(&f, P);
        let best = feats.iter().map(|r| r[0]).fold(0.0, f32::max);
        assert!(best >= potential_chain(&f).chains as f32 / 10.0);
    }
}
