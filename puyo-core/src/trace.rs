//! 連鎖の中間盤面の記録(リプレイ・動画用)。
//!
//! 進行ロジックは [`Tracer`] を受け取るジェネリック関数として1本だけ書いてあり、
//! 通常の `step` は何もしない [`NoTrace`] を渡す(最適化で消える)。
//! そのため `step` と `step_traced` の結果が食い違うことはない。

use crate::field::Field;

/// フレームの種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameKind {
    /// 組ぷよを置いて落下させた直後(連鎖前)。
    Placed,
    /// 消去の直前。`cleared` に消えるセルのマスクが入る。
    BeforeClear,
    /// 消去の直後(浮いたぷよが残っている)。
    AfterClear,
    /// 落下の後。
    AfterGravity,
    /// おじゃまが降った後(対戦のみ)。
    OjamaDropped,
}

impl FrameKind {
    pub fn name(self) -> &'static str {
        match self {
            FrameKind::Placed => "placed",
            FrameKind::BeforeClear => "before_clear",
            FrameKind::AfterClear => "after_clear",
            FrameKind::AfterGravity => "after_gravity",
            FrameKind::OjamaDropped => "ojama_dropped",
        }
    }
}

/// 1枚の中間盤面。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceFrame {
    pub kind: FrameKind,
    pub field: Field,
    /// その時点の連鎖数(0 = 連鎖前)。
    pub chain: u32,
    /// `BeforeClear` で消えるセルのマスク(`Field` のビット配置)。それ以外は0。
    pub cleared: u128,
    /// その時点の累計得点。
    pub score: u64,
}

/// 中間盤面の受け取り先。
pub trait Tracer {
    fn frame(&mut self, kind: FrameKind, field: &Field, chain: u32, cleared: u128, score: u64);
}

/// 何も記録しない。
pub struct NoTrace;

impl Tracer for NoTrace {
    #[inline(always)]
    fn frame(&mut self, _: FrameKind, _: &Field, _: u32, _: u128, _: u64) {}
}

impl Tracer for Vec<TraceFrame> {
    fn frame(&mut self, kind: FrameKind, field: &Field, chain: u32, cleared: u128, score: u64) {
        self.push(TraceFrame {
            kind,
            field: *field,
            chain,
            cleared,
            score,
        });
    }
}
