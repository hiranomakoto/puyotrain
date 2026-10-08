//! 得点からおじゃまへの換算と相殺。

use crate::rules::{ALL_CLEAR_BONUS_OJAMA, SCORE_PER_OJAMA};

/// 得点をおじゃま数に換算する。端数は `carry` に持ち越す。
#[inline]
pub fn score_to_ojama(score: u32, carry: &mut u32) -> u32 {
    let total = score + *carry;
    *carry = total % SCORE_PER_OJAMA;
    total / SCORE_PER_OJAMA
}

/// 連鎖1回分の攻撃おじゃま数(全消しボーナス込み)。
#[inline]
pub fn attack_ojama(score: u32, carry: &mut u32, all_clear_bonus: bool) -> u32 {
    let mut n = score_to_ojama(score, carry);
    if all_clear_bonus {
        n += ALL_CLEAR_BONUS_OJAMA;
    }
    n
}

/// 自分の予告おじゃまを攻撃で相殺する。残った攻撃数を返す。
#[inline]
pub fn offset(attack: u32, pending: &mut u32) -> u32 {
    let o = attack.min(*pending);
    *pending -= o;
    attack - o
}
