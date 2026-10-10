//! 手ごとの結果の特徴量。22通りの手それぞれについて、置いた後の盤面を実際に計算して
//! 「その手を打つと何が起きるか」を数値にする。方策ネットが手ごとに点数をつけるのに使う。

use crate::action::{legal_actions, place_pair, Action};
use crate::chain::{resolve_chain, ChainSummary};
use crate::color::Pair;
use crate::field::Field;
use crate::potential::potential_chain;
use crate::rules::{DEATH_COL, HEIGHT, NUM_ACTIONS, VISIBLE_HEIGHT, WIDTH};

/// 1手あたりの特徴量の数。
pub const NUM_ACTION_FEATURES: usize = 13;

/// 特徴量の名前(並び順)。
pub const ACTION_FEATURE_NAMES: [&str; NUM_ACTION_FEATURES] = [
    "legal",
    "dead",
    "chains",
    "log_score",
    "potential_chains",
    "potential_log_score",
    "max_height",
    "death_col_height",
    "count",
    "adjacency",
    "survivable",
    "next_best_chains",
    "next_best_log_score",
];

/// 得点を 0〜1 程度に収める(ln(1 + 10万点) ≒ 11.5)。
#[inline]
fn log_score(score: u32) -> f32 {
    (score as f32).ln_1p() / 12.0
}

#[inline]
fn simulate(field: &Field, pair: Pair, action: Action) -> (Field, ChainSummary) {
    let mut f = *field;
    place_pair(&mut f, pair, action);
    let s = resolve_chain(&mut f);
    (f, s)
}

/// 手ごとの特徴量。合法でない手は全部0。
///
/// - legal: 合法手なら1
/// - dead: 置いた結果窒息するなら1
/// - chains, log_score: この手で起きる連鎖数(/10)と得点(対数)
/// - potential_chains, potential_log_score: 置いた後の潜在連鎖(/10、得点は対数)。窒息なら0
/// - max_height, death_col_height: 置いた後の最大の高さと窒息列の高さ(/13)
/// - count: 置いた後のぷよの数(/72)
/// - adjacency: 置いた後の同色の隣接数(/100)
/// - survivable: ネクストを窒息せずに置ける手があるなら1
/// - next_best_chains, next_best_log_score: 置いた後、ネクストで起こせる最大の連鎖(/10)と得点(対数)
pub fn action_features(
    field: &Field,
    pairs: [Pair; 2],
) -> [[f32; NUM_ACTION_FEATURES]; NUM_ACTIONS] {
    let mut out = [[0.0; NUM_ACTION_FEATURES]; NUM_ACTIONS];
    let mut m1 = legal_actions(field);
    while m1 != 0 {
        let a1 = Action::new(m1.trailing_zeros() as usize).unwrap();
        m1 &= m1 - 1;
        let (f1, s1) = simulate(field, pairs[0], a1);
        let dead = f1.is_dead();
        let heights = f1.heights();
        let row = &mut out[a1.index()];
        row[0] = 1.0;
        row[1] = dead as u8 as f32;
        row[2] = s1.chains as f32 / 10.0;
        row[3] = log_score(s1.score);
        row[6] = *heights.iter().max().unwrap() as f32 / HEIGHT as f32;
        row[7] = heights[DEATH_COL] as f32 / HEIGHT as f32;
        row[8] = f1.count() as f32 / (VISIBLE_HEIGHT * WIDTH) as f32;
        row[9] = f1.same_color_adjacency() as f32 / 100.0;
        if dead {
            continue;
        }
        let p = potential_chain(&f1);
        row[4] = p.chains as f32 / 10.0;
        row[5] = log_score(p.score);

        let mut survivable = false;
        let mut best = (0u32, 0u32);
        let mut m2 = legal_actions(&f1);
        while m2 != 0 {
            let a2 = Action::new(m2.trailing_zeros() as usize).unwrap();
            m2 &= m2 - 1;
            let (f2, s2) = simulate(&f1, pairs[1], a2);
            survivable |= !f2.is_dead();
            best = best.max((s2.chains, s2.score));
        }
        row[10] = survivable as u8 as f32;
        row[11] = best.0 as f32 / 10.0;
        row[12] = log_score(best.1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Puyo;

    const RG: Pair = Pair::new(Puyo::Red, Puyo::Green);
    const BY: Pair = Pair::new(Puyo::Blue, Puyo::Yellow);

    #[test]
    fn illegal_actions_are_zero_and_legal_flag_matches() {
        let f = Field::new();
        let feats = action_features(&f, [RG, BY]);
        let legal = legal_actions(&f);
        for (i, row) in feats.iter().enumerate() {
            assert_eq!(row[0] == 1.0, legal >> i & 1 == 1);
            if legal >> i & 1 == 0 {
                assert!(row.iter().all(|&x| x == 0.0));
            } else {
                assert_eq!(row[8], 2.0 / 72.0);
                assert_eq!(row[10], 1.0);
            }
        }
    }

    #[test]
    fn clearing_move_reports_chain() {
        // 列0 に R が3個。R を下にした縦置きで列0に置くと消える
        let f = Field::from_rows(&["R.....", "R.....", "R....."]).unwrap();
        let feats = action_features(&f, [RG, BY]);
        let best = feats.iter().map(|r| r[2]).fold(0.0, f32::max);
        assert_eq!(best, 0.1);
    }

    #[test]
    fn potential_after_placement() {
        // 列0 に R を2個置く手は潜在連鎖1(もう1個 R を足せば消える)を作る
        let f = Field::from_rows(&["R.....", "R....."]).unwrap();
        let feats = action_features(&f, [Pair::new(Puyo::Green, Puyo::Green), BY]);
        assert!(feats.iter().any(|r| r[4] == 0.1));
    }
}
