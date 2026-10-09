//! ルールベースのボット(ベースライン用)。
//!
//! 2手読み: 現在の組ぷよとネクストの置き方 22×22 通りを全部試し、
//! 2手の合計得点が最大になる1手目を選ぶ。同点の場合は順に
//! 1. 窒息しない
//! 2. 同色の隣接数が多い
//! 3. 最大の高さが低い
//! 4. それでも同点なら乱数
//!
//! で決める。

use crate::action::{legal_actions, place_pair, Action};
use crate::chain::resolve_chain;
use crate::color::Pair;
use crate::field::Field;
use crate::tsumo::Rng;

/// 評価値(辞書式に大きいほど良い)。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    score: u32,
    alive: bool,
    adjacency: u32,
    neg_height: i32,
}

fn leaf_key(field: &Field, score: u32) -> Key {
    let max_h = field.heights().into_iter().max().unwrap_or(0);
    Key {
        score,
        alive: !field.is_dead(),
        adjacency: field.same_color_adjacency(),
        neg_height: -(max_h as i32),
    }
}

/// 1手置いた結果の盤面と得点。
#[inline]
fn simulate(field: &Field, pair: Pair, action: Action) -> (Field, u32) {
    let mut f = *field;
    place_pair(&mut f, pair, action);
    let s = resolve_chain(&mut f).score;
    (f, s)
}

/// 2手読みで1手目を選ぶ。合法手が無ければ `None`。
pub fn greedy_action(field: &Field, pairs: [Pair; 2], rng: &mut Rng) -> Option<Action> {
    let mut best: Option<(Key, Action)> = None;
    let mut ties = 0u32;
    let mut m1 = legal_actions(field);
    while m1 != 0 {
        let a1 = Action::new(m1.trailing_zeros() as usize).unwrap();
        m1 &= m1 - 1;
        let (f1, s1) = simulate(field, pairs[0], a1);
        let key = if f1.is_dead() {
            leaf_key(&f1, s1)
        } else {
            let mut m2 = legal_actions(&f1);
            let mut k2: Option<Key> = None;
            while m2 != 0 {
                let a2 = Action::new(m2.trailing_zeros() as usize).unwrap();
                m2 &= m2 - 1;
                let (f2, s2) = simulate(&f1, pairs[1], a2);
                let k = leaf_key(&f2, s1 + s2);
                if k2.is_none_or(|b| k > b) {
                    k2 = Some(k);
                }
            }
            k2.unwrap_or_else(|| leaf_key(&f1, s1))
        };
        match best {
            Some((bk, _)) if key < bk => {}
            Some((bk, _)) if key == bk => {
                // 同点はリザーバサンプリングで一様に選ぶ
                ties += 1;
                if rng.below(ties + 1) == 0 {
                    best = Some((key, a1));
                }
            }
            _ => {
                best = Some((key, a1));
                ties = 0;
            }
        }
    }
    best.map(|(_, a)| a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Puyo;
    use crate::Game;

    #[test]
    fn takes_obvious_clear() {
        // 列0 に R が3個。R を含む組ぷよなら列0に置いて消すはず。
        let f = Field::from_rows(&["R.....", "R.....", "R....."]).unwrap();
        let pairs = [
            Pair::new(Puyo::Red, Puyo::Green),
            Pair::new(Puyo::Blue, Puyo::Yellow),
        ];
        let a = greedy_action(&f, pairs, &mut Rng::new(0)).unwrap();
        let (_, s) = simulate(&f, pairs[0], a);
        assert!(s >= 40);
    }

    #[test]
    fn greedy_beats_random_on_average() {
        let mut greedy_total = 0u64;
        let mut random_total = 0u64;
        for seed in 0..20 {
            let mut g = Game::new(seed);
            let mut rng = Rng::new(seed);
            while !g.is_over() && g.turn() < 200 {
                let a = greedy_action(g.field(), [g.pair(0), g.pair(1)], &mut rng).unwrap();
                g.step(a).unwrap();
            }
            greedy_total += g.score();
            let mut g = Game::new(seed);
            while !g.is_over() && g.turn() < 200 {
                let m = g.legal_actions();
                let n = rng.below(m.count_ones());
                let mut mm = m;
                for _ in 0..n {
                    mm &= mm - 1;
                }
                g.step(Action::new(mm.trailing_zeros() as usize).unwrap())
                    .unwrap();
            }
            random_total += g.score();
        }
        assert!(
            greedy_total > random_total * 2,
            "{greedy_total} vs {random_total}"
        );
    }
}
