//! 潜在連鎖: どこかの列に同じ色のぷよを1〜2個足したとき、最大で何連鎖が起きるか。
//!
//! 盤面に仕込まれた連鎖の「種」の大きさの目安。学習の報酬整形と評価の指標に使う。
//! 足すぷよは列の上に積むだけで、そこまで操作で届くかは考えない。

use crate::chain::resolve_chain;
use crate::color::Puyo;
use crate::field::{Field, VISIBLE_MASK};
use crate::rules::{VISIBLE_HEIGHT, WIDTH};

/// 1列に足すぷよの最大数。
pub const MAX_ADDED: usize = 2;

/// 潜在連鎖の大きさ。連鎖数、同じなら得点で比べる。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Potential {
    pub chains: u32,
    pub score: u32,
}

/// (col, row) の上下左右に `plane` のぷよがあるか(12段目まで)。
#[inline]
fn touches(plane: u128, col: usize, row: usize) -> bool {
    let b = 1u128 << (col * 16 + row);
    plane & ((b << 1) | (b >> 1) | (b << 16) | (b >> 16)) & VISIBLE_MASK != 0
}

/// 6列 × 4色 × 1〜2個を試し、起きる連鎖が最大のものを返す。
///
/// 足したぷよが同じ色のぷよに接していなければ消えないので、その場合は連鎖を計算しない
/// (結果は全部計算した場合と同じ)。
pub fn potential_chain(field: &Field) -> Potential {
    let mut best = Potential::default();
    for col in 0..WIDTH {
        // 13段目のぷよは連結しないので、12段目までに置ける場合だけ
        let h = field.height(col);
        if h >= VISIBLE_HEIGHT {
            continue;
        }
        let room2 = h + 1 < VISIBLE_HEIGHT;
        for color in Puyo::COLORS {
            let plane = field.plane(color);
            let t1 = touches(plane, col, h);
            let t2 = room2 && touches(plane, col, h + 1);
            if !t1 && !t2 {
                continue;
            }
            let mut f = *field;
            f.push(col, color);
            if t1 {
                let mut g = f;
                let s = resolve_chain(&mut g);
                if s.chains > 0 {
                    // 1個目で発火したら、それ以上足しても同じ連鎖になる
                    best = best.max(Potential {
                        chains: s.chains,
                        score: s.score,
                    });
                    continue;
                }
            }
            if room2 {
                f.push(col, color);
                let s = resolve_chain(&mut f);
                best = best.max(Potential {
                    chains: s.chains,
                    score: s.score,
                });
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_field_has_no_potential() {
        assert_eq!(potential_chain(&Field::new()), Potential::default());
    }

    #[test]
    fn three_in_a_column_is_one_chain() {
        let f = Field::from_rows(&["R.....", "R.....", "R....."]).unwrap();
        assert_eq!(potential_chain(&f).chains, 1);
    }

    #[test]
    fn two_puyos_need_two_added() {
        let f = Field::from_rows(&["R.....", "R....."]).unwrap();
        assert_eq!(potential_chain(&f).chains, 1);
    }

    #[test]
    fn finds_two_chain() {
        // 列1の上に R を足すと R が4個消え、列0の G が落ちて G が4個つながる
        let f = Field::from_rows(&["G.....", "RR....", "RGGG.."]).unwrap();
        let mut g = f;
        assert_eq!(resolve_chain(&mut g).chains, 0);
        assert_eq!(potential_chain(&f).chains, 2);
    }

    /// 枝刈りなしで全部計算する版(比較用)。
    fn brute_force(field: &Field) -> Potential {
        let mut best = Potential::default();
        for col in 0..WIDTH {
            for color in Puyo::COLORS {
                let mut f = *field;
                for _ in 0..MAX_ADDED {
                    if f.height(col) >= VISIBLE_HEIGHT || !f.push(col, color) {
                        break;
                    }
                    let mut g = f;
                    let s = resolve_chain(&mut g);
                    best = best.max(Potential {
                        chains: s.chains,
                        score: s.score,
                    });
                    if s.chains > 0 {
                        break;
                    }
                }
            }
        }
        best
    }

    #[test]
    fn pruning_matches_brute_force_on_played_fields() {
        let mut rng = crate::tsumo::Rng::new(7);
        let mut checked = 0;
        for seed in 0..40 {
            let mut g = crate::Game::new(seed);
            while !g.is_over() && g.turn() < 120 {
                assert_eq!(potential_chain(g.field()), brute_force(g.field()));
                checked += 1;
                // 半分は2手読み、半分はランダムで、いろいろな形の盤面を作る
                let a = if rng.below(2) == 0 {
                    g.greedy_action(&mut rng).unwrap()
                } else {
                    let m = g.legal_actions();
                    let mut mm = m;
                    for _ in 0..rng.below(m.count_ones()) {
                        mm &= mm - 1;
                    }
                    crate::Action::new(mm.trailing_zeros() as usize).unwrap()
                };
                g.step(a).unwrap();
            }
        }
        assert!(checked > 1000);
    }

    #[test]
    fn does_not_modify_field() {
        let f = Field::from_rows(&["G.....", "RR....", "RGGG.."]).unwrap();
        let before = f;
        potential_chain(&f);
        assert_eq!(f, before);
    }
}
