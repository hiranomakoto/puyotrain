//! 1プレイヤー分の状態と、ターン進行の部品(設置 → 連鎖 → おじゃま授受 → 降下 → 窒息判定)。
//!
//! ここにある関数はどれも「いつ呼ぶか」を決めない。呼ぶ順番とタイミングは
//! `Game` / `Versus` が決める(現在は同期ターン制)。将来フレーム時間モデルを足すときは、
//! 同じ部品を時間付きで呼ぶ別のスケジューラを書けばよい。

use crate::action::{place_pair, Action};
use crate::chain::{resolve_chain, ChainSummary};
use crate::color::Pair;
use crate::field::Field;
use crate::rules::{MAX_OJAMA_DROP, WIDTH};
use crate::score::{attack_ojama, offset};
use crate::tsumo::Rng;

/// 1プレイヤー分の状態。`Copy` なので clone が安い。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlayerState {
    pub field: Field,
    /// 予告おじゃま数。
    pub pending: u32,
    /// おじゃま換算の端数(得点)。
    pub carry: u32,
    /// 全消しボーナス待ち(次の連鎖で30個上乗せ)。
    pub all_clear: bool,
    /// 累計得点。
    pub score: u64,
    /// おじゃまの落下位置を決める乱数。
    pub ojama_rng: Rng,
}

impl PlayerState {
    pub fn new(ojama_seed: u64) -> PlayerState {
        PlayerState {
            field: Field::new(),
            pending: 0,
            carry: 0,
            all_clear: false,
            score: 0,
            ojama_rng: Rng::new(ojama_seed),
        }
    }
}

/// 設置と連鎖の結果。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Placement {
    pub chain: ChainSummary,
    /// この連鎖で生まれた攻撃おじゃま数(全消しボーナス込み、相殺前)。
    pub attack: u32,
    /// この手で全消しになったか。
    pub all_clear: bool,
}

/// 組ぷよを置き、連鎖を最後まで解決する。
/// 連鎖が起きたら得点をおじゃまに換算し、全消しボーナスを消費する。
pub fn place_and_resolve(st: &mut PlayerState, pair: Pair, action: Action) -> Placement {
    place_pair(&mut st.field, pair, action);
    let chain = resolve_chain(&mut st.field);
    let mut out = Placement {
        chain,
        ..Placement::default()
    };
    if chain.chains > 0 {
        st.score += chain.score as u64;
        out.attack = attack_ojama(chain.score, &mut st.carry, st.all_clear);
        st.all_clear = false;
        if st.field.is_empty() {
            st.all_clear = true;
            out.all_clear = true;
        }
    }
    out
}

/// おじゃま `n` 個(30個以下)の列ごとの配分。
/// 6個ごとに1段ずつ全列、端数は重複なしのランダムな列に1個ずつ。
pub fn ojama_pattern(n: u32, rng: &mut Rng) -> [u32; WIDTH] {
    let w = WIDTH as u32;
    let mut counts = [n / w; WIDTH];
    let rem = (n % w) as usize;
    if rem > 0 {
        let mut cols: [usize; WIDTH] = [0, 1, 2, 3, 4, 5];
        for i in 0..rem {
            let j = i + rng.below((WIDTH - i) as u32) as usize;
            cols.swap(i, j);
            counts[cols[i]] += 1;
        }
    }
    counts
}

/// 予告おじゃまを最大30個降らせる。降らせた個数を返す。
pub fn fall_ojama(st: &mut PlayerState) -> u32 {
    let n = st.pending.min(MAX_OJAMA_DROP);
    if n == 0 {
        return 0;
    }
    let counts = ojama_pattern(n, &mut st.ojama_rng);
    st.field.drop_ojama_columns(&counts);
    st.pending -= n;
    n
}

/// 同時に発生した攻撃を授受する。
///
/// 1. 各自の攻撃でまず自分の予告おじゃまを相殺する。
/// 2. 残った攻撃同士を打ち消し合い、差分を相手の予告に加える。
///
/// 戻り値は相手の予告に加わった個数。
pub fn exchange(players: &mut [PlayerState; 2], attacks: [u32; 2]) -> [u32; 2] {
    let a0 = offset(attacks[0], &mut players[0].pending);
    let a1 = offset(attacks[1], &mut players[1].pending);
    let mut sent = [0; 2];
    if a0 > a1 {
        sent[0] = a0 - a1;
        players[1].pending += sent[0];
    } else if a1 > a0 {
        sent[1] = a1 - a0;
        players[0].pending += sent[1];
    }
    sent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_sums() {
        let mut rng = Rng::new(3);
        for n in 0..=30 {
            let p = ojama_pattern(n, &mut rng);
            assert_eq!(p.iter().sum::<u32>(), n);
            let lo = n / 6;
            assert!(p.iter().all(|&c| c == lo || c == lo + 1));
        }
    }
}
