//! ぷよぷよ通ルール準拠の対戦シミュレータ。
//!
//! - [`Field`] — 盤面(ビットボード、`Copy`)
//! - [`Game`] — 1人用
//! - [`Versus`] — 2人対戦(同期ターン制)
//!
//! ルール定数は [`rules`] に集約している。

#![forbid(unsafe_code)]

pub mod action;
pub mod afterstate;
pub mod bot;
pub mod chain;
pub mod color;
pub mod field;
pub mod game;
pub mod phase;
pub mod potential;
pub mod rules;
pub mod score;
pub mod trace;
pub mod tsumo;
pub mod versus;

pub use action::{legal_actions, place_pair, Action, ActionMask, Rotation};
pub use chain::{resolve_chain, step_score, ChainResolver, ChainStep, ChainSummary};
pub use color::{Pair, Puyo};
pub use field::{ClearResult, Field};
pub use game::{Game, GameStep, StepError};
pub use phase::{Placement, PlayerState};
pub use potential::{potential_chain, Potential};
pub use trace::{FrameKind, NoTrace, TraceFrame, Tracer};
pub use tsumo::{Rng, Tsumo};
pub use versus::{Outcome, PlayerTurn, Versus, VersusStep};
