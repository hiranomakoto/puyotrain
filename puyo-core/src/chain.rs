//! 連鎖の解決と得点計算。
//!
//! `ChainResolver` は1連鎖ずつ進めるイテレータ。同期ターン制では一気に回し切るが、
//! 将来のフレーム時間モデルでは1ステップごとに時間を割り当てられる。

use crate::field::{ClearResult, Field};
use crate::rules::{self, SCORE_PER_PUYO};

/// 1連鎖分の結果。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainStep {
    /// 何連鎖目か(1始まり)。
    pub chain: u32,
    pub clear: ClearResult,
    /// この連鎖で得た得点。
    pub score: u32,
}

/// 1回の消去の得点: 消去数 × 10 × (連鎖 + 連結 + 色数ボーナス、最小1)。
#[inline]
pub fn step_score(chain: u32, clear: &ClearResult) -> u32 {
    let bonus = rules::chain_bonus(chain) + clear.link_bonus + rules::color_bonus(clear.num_colors);
    clear.num_colored * SCORE_PER_PUYO * bonus.max(1)
}

/// 盤面の連鎖を1ステップずつ解決する。
pub struct ChainResolver<'a> {
    field: &'a mut Field,
    chain: u32,
}

impl<'a> ChainResolver<'a> {
    /// 落下済みの盤面を受け取る。
    pub fn new(field: &'a mut Field) -> Self {
        ChainResolver { field, chain: 0 }
    }
}

impl Iterator for ChainResolver<'_> {
    type Item = ChainStep;

    fn next(&mut self) -> Option<ChainStep> {
        let clear = self.field.clear_groups()?;
        self.field.apply_gravity();
        self.chain += 1;
        Some(ChainStep {
            chain: self.chain,
            score: step_score(self.chain, &clear),
            clear,
        })
    }
}

/// 連鎖全体の集計。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChainSummary {
    /// 連鎖数(0なら連鎖なし)。
    pub chains: u32,
    pub score: u32,
    /// 消えた色ぷよの総数。
    pub cleared: u32,
    /// 消えたおじゃまの総数。
    pub ojama_cleared: u32,
}

/// 連鎖を最後まで解決する。
pub fn resolve_chain(field: &mut Field) -> ChainSummary {
    let mut s = ChainSummary::default();
    for step in ChainResolver::new(field) {
        s.chains = step.chain;
        s.score += step.score;
        s.cleared += step.clear.num_colored;
        s.ojama_cleared += step.clear.num_ojama;
    }
    s
}
