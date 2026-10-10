//! 1人用ゲーム。

use std::fmt;

use crate::action::{legal_actions, Action, ActionMask};
use crate::color::Pair;
use crate::field::Field;
use crate::phase::{place_and_resolve_with, Placement, PlayerState};
use crate::trace::{NoTrace, TraceFrame, Tracer};
use crate::tsumo::{mix64, Tsumo};

/// `step` が受け付けられなかった理由。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepError {
    /// 合法手でない(Versus では何番目のプレイヤーか)。
    IllegalAction { player: usize, action: Action },
    /// すでに終了している。
    GameOver,
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepError::IllegalAction { player, action } => {
                write!(f, "illegal action {} by player {}", action.index(), player)
            }
            StepError::GameOver => write!(f, "game is already over"),
        }
    }
}

impl std::error::Error for StepError {}

/// 1手の結果。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameStep {
    pub placement: Placement,
    /// この手で窒息したか。
    pub dead: bool,
}

/// 1人用ゲーム。`Copy` なので clone が安い。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Game {
    player: PlayerState,
    tsumo: Tsumo,
    turn: u64,
    dead: bool,
}

/// プレイヤーごとのおじゃま乱数シード。
pub(crate) fn ojama_seed(seed: u64, player: usize) -> u64 {
    mix64(seed ^ 0x6f6a_616d_6100_0000 ^ player as u64)
}

impl Game {
    pub fn new(seed: u64) -> Game {
        Game {
            player: PlayerState::new(ojama_seed(seed, 0)),
            tsumo: Tsumo::new(seed),
            turn: 0,
            dead: false,
        }
    }

    pub fn field(&self) -> &Field {
        &self.player.field
    }

    pub fn player(&self) -> &PlayerState {
        &self.player
    }

    pub fn score(&self) -> u64 {
        self.player.score
    }

    /// 置いた手数。
    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn is_over(&self) -> bool {
        self.dead
    }

    pub fn seed(&self) -> u64 {
        self.tsumo.seed()
    }

    /// 現在の組ぷよ(k=0)、ネクスト(k=1)、ネクネク(k=2)。
    pub fn pair(&self, k: u64) -> Pair {
        self.tsumo.pair(self.turn + k)
    }

    /// 現在・ネクスト・ネクネクの3組。
    pub fn visible_pairs(&self) -> [Pair; 3] {
        [self.pair(0), self.pair(1), self.pair(2)]
    }

    /// ルールベース(2手読み)の手。終了後は `None`。
    pub fn greedy_action(&self, rng: &mut crate::tsumo::Rng) -> Option<Action> {
        if self.dead {
            return None;
        }
        crate::bot::greedy_action(&self.player.field, [self.pair(0), self.pair(1)], rng)
    }

    /// 自滅しない手のマスク([`crate::bot::safe_actions`])。終了後は0。
    pub fn safe_actions(&self) -> ActionMask {
        if self.dead {
            0
        } else {
            crate::bot::safe_actions(&self.player.field, [self.pair(0), self.pair(1)])
        }
    }

    /// 手ごとの結果の特徴量([`crate::afterstate::action_features`])。終了後は全部0。
    pub fn action_features(
        &self,
    ) -> [[f32; crate::afterstate::NUM_ACTION_FEATURES]; crate::rules::NUM_ACTIONS] {
        if self.dead {
            [[0.0; crate::afterstate::NUM_ACTION_FEATURES]; crate::rules::NUM_ACTIONS]
        } else {
            crate::afterstate::action_features(&self.player.field, [self.pair(0), self.pair(1)])
        }
    }

    /// 現在の盤面の潜在連鎖([`crate::potential::potential_chain`])。
    pub fn potential_chain(&self) -> crate::potential::Potential {
        crate::potential::potential_chain(&self.player.field)
    }

    /// 合法手マスク(終了後は0)。
    pub fn legal_actions(&self) -> ActionMask {
        if self.dead {
            0
        } else {
            legal_actions(&self.player.field)
        }
    }

    pub fn step(&mut self, action: Action) -> Result<GameStep, StepError> {
        self.step_with(action, &mut NoTrace)
    }

    /// `step` と同じ。設置後と連鎖の中間盤面(消去前・消去後・落下後)も返す。
    pub fn step_traced(
        &mut self,
        action: Action,
    ) -> Result<(GameStep, Vec<TraceFrame>), StepError> {
        let mut frames = Vec::new();
        let r = self.step_with(action, &mut frames)?;
        Ok((r, frames))
    }

    pub fn step_with<T: Tracer>(
        &mut self,
        action: Action,
        tracer: &mut T,
    ) -> Result<GameStep, StepError> {
        if self.dead {
            return Err(StepError::GameOver);
        }
        if legal_actions(&self.player.field) & (1 << action.index()) == 0 {
            return Err(StepError::IllegalAction { player: 0, action });
        }
        let pair = self.tsumo.pair(self.turn);
        let placement = place_and_resolve_with(&mut self.player, pair, action, tracer);
        self.turn += 1;
        self.dead = self.player.field.is_dead();
        Ok(GameStep {
            placement,
            dead: self.dead,
        })
    }
}
