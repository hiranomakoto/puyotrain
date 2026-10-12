//! 先読みで連鎖を組む探索(試作)。
//!
//! 見えている組ぷよ(現在・ネクスト・ネクネク)の置き方を `depth` 手先まで全部試し、
//! 評価が最大の並びの最初の1手を返す。
//!
//! - 組む(`fire = false`): 並びの最後の盤面の潜在連鎖が大きいほど良い。途中で連鎖を撃つと
//!   種が崩れるので、自然に避けられる。
//! - 撃つ(`fire = true`): 並びの途中で撃てる連鎖の最大が大きいほど良い(同じなら得点、
//!   最後の潜在連鎖)。残りの手数が少ないときに使う。
//!
//! どちらも窒息する並びは選ばない(全部窒息するなら合法手のどれか)。

use crate::action::{legal_actions, place_pair, Action};
use crate::chain::resolve_chain;
use crate::color::Pair;
use crate::field::Field;
use crate::potential::potential_chain;

/// 並びの評価(辞書式に大きいほど良い)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Score {
    alive: bool,
    primary: u32,
    secondary: u32,
    tertiary: u32,
}

/// `pairs[k..]` を置いていく並びの最良の評価。`fired` はここまでに撃った最大の連鎖と得点。
fn search(field: &Field, pairs: &[Pair], fire: bool, fired: (u32, u32)) -> Score {
    if pairs.is_empty() {
        let pot = potential_chain(field);
        return if fire {
            Score {
                alive: true,
                primary: fired.0,
                secondary: fired.1,
                tertiary: pot.chains,
            }
        } else {
            Score {
                alive: true,
                primary: pot.chains,
                secondary: pot.score,
                tertiary: 0,
            }
        };
    }
    let mut best = Score::default();
    let mut m = legal_actions(field);
    while m != 0 {
        let a = Action::new(m.trailing_zeros() as usize).unwrap();
        m &= m - 1;
        let mut f = *field;
        place_pair(&mut f, pairs[0], a);
        let s = resolve_chain(&mut f);
        if f.is_dead() {
            continue;
        }
        let fired = fired.max((s.chains, s.score));
        best = best.max(search(&f, &pairs[1..], fire, fired));
    }
    best
}

/// 先読みで選んだ手。`pairs` は現在から順の組ぷよ(`depth` 個まで使う)。合法手が無ければ None。
pub fn builder_action(field: &Field, pairs: &[Pair], depth: usize, fire: bool) -> Option<Action> {
    let pairs = &pairs[..depth.clamp(1, pairs.len())];
    let mut best: Option<(Score, Action)> = None;
    let mut m = legal_actions(field);
    while m != 0 {
        let a = Action::new(m.trailing_zeros() as usize).unwrap();
        m &= m - 1;
        let mut f = *field;
        place_pair(&mut f, pairs[0], a);
        let s = resolve_chain(&mut f);
        let score = if f.is_dead() {
            Score::default()
        } else {
            search(&f, &pairs[1..], fire, (s.chains, s.score))
        };
        if best.is_none_or(|(b, _)| score > b) {
            best = Some((score, a));
        }
    }
    best.map(|(_, a)| a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Puyo;

    #[test]
    fn fire_mode_takes_the_available_chain() {
        // 列0 に R が3個。R を含む組ぷよなら撃てる
        let f = Field::from_rows(&["R.....", "R.....", "R....."]).unwrap();
        let pairs = [Pair::new(Puyo::Red, Puyo::Green)];
        let a = builder_action(&f, &pairs, 1, true).unwrap();
        let mut g = f;
        place_pair(&mut g, pairs[0], a);
        assert_eq!(resolve_chain(&mut g).chains, 1);
    }

    #[test]
    fn build_mode_keeps_the_seed() {
        // 列0 に R が3個。組む段階では消さずに残す(撃つと潜在連鎖が減る)
        let f = Field::from_rows(&["R.....", "R.....", "R....."]).unwrap();
        let pairs = [
            Pair::new(Puyo::Red, Puyo::Red),
            Pair::new(Puyo::Green, Puyo::Blue),
        ];
        let a = builder_action(&f, &pairs, 2, false).unwrap();
        let mut g = f;
        place_pair(&mut g, pairs[0], a);
        assert_eq!(resolve_chain(&mut g).chains, 0);
    }
}
