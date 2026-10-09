//! 2人対戦(同期ターン制)。
//!
//! 1ターンの進行:
//! 1. 両者が同じ組ぷよを同時に置き、それぞれ連鎖を解決する。
//! 2. このターン連鎖しなかったプレイヤーには、前のターンまでに届いていた予告おじゃまが
//!    最大30個降る(このターンに届く分はまだ降らない = 1手の猶予)。
//! 3. 攻撃を授受する(自分の予告を相殺 → 残りを打ち消し合い → 差分を相手の予告へ)。
//! 4. 窒息判定。両者同時なら引き分け。

use crate::action::{legal_actions, Action, ActionMask};
use crate::color::Pair;
use crate::game::{ojama_seed, StepError};
use crate::phase::{exchange, fall_ojama, place_and_resolve_with, Placement, PlayerState};
use crate::trace::{FrameKind, NoTrace, TraceFrame, Tracer};
use crate::tsumo::Tsumo;

/// 対戦の結果。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// 勝者のプレイヤー番号(0 または 1)。
    Win(usize),
    Draw,
}

/// 1ターン分の各プレイヤーの結果。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlayerTurn {
    pub placement: Placement,
    /// このターンに降ったおじゃま数。
    pub ojama_dropped: u32,
    /// 相殺後、相手の予告に送った個数。
    pub ojama_sent: u32,
    pub dead: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VersusStep {
    pub players: [PlayerTurn; 2],
    /// 決着したら `Some`。
    pub outcome: Option<Outcome>,
}

/// 2人対戦。`Copy` なので clone が安い。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Versus {
    players: [PlayerState; 2],
    tsumo: Tsumo,
    turn: u64,
    outcome: Option<Outcome>,
}

impl Versus {
    pub fn new(seed: u64) -> Versus {
        Versus {
            players: [
                PlayerState::new(ojama_seed(seed, 0)),
                PlayerState::new(ojama_seed(seed, 1)),
            ],
            tsumo: Tsumo::new(seed),
            turn: 0,
            outcome: None,
        }
    }

    pub fn player(&self, i: usize) -> &PlayerState {
        &self.players[i]
    }

    pub fn players(&self) -> &[PlayerState; 2] {
        &self.players
    }

    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    pub fn is_over(&self) -> bool {
        self.outcome.is_some()
    }

    pub fn seed(&self) -> u64 {
        self.tsumo.seed()
    }

    /// 現在の組ぷよ(k=0)、ネクスト(k=1)、ネクネク(k=2)。両者共通。
    pub fn pair(&self, k: u64) -> Pair {
        self.tsumo.pair(self.turn + k)
    }

    pub fn visible_pairs(&self) -> [Pair; 3] {
        [self.pair(0), self.pair(1), self.pair(2)]
    }

    /// プレイヤー `i` のルールベース(2手読み)の手。終了後は `None`。
    pub fn greedy_action(&self, i: usize, rng: &mut crate::tsumo::Rng) -> Option<Action> {
        if self.is_over() {
            return None;
        }
        crate::bot::greedy_action(&self.players[i].field, [self.pair(0), self.pair(1)], rng)
    }

    /// プレイヤー `i` の合法手マスク(終了後は0)。
    pub fn legal_actions(&self, i: usize) -> ActionMask {
        if self.is_over() {
            0
        } else {
            legal_actions(&self.players[i].field)
        }
    }

    pub fn step(&mut self, a0: Action, a1: Action) -> Result<VersusStep, StepError> {
        self.step_with(a0, a1, &mut [NoTrace, NoTrace])
    }

    /// `step` と同じ。各プレイヤーの中間盤面(設置後・連鎖・おじゃま降下後)も返す。
    #[allow(clippy::type_complexity)]
    pub fn step_traced(
        &mut self,
        a0: Action,
        a1: Action,
    ) -> Result<(VersusStep, [Vec<TraceFrame>; 2]), StepError> {
        let mut frames = [Vec::new(), Vec::new()];
        let r = self.step_with(a0, a1, &mut frames)?;
        Ok((r, frames))
    }

    pub fn step_with<T: Tracer>(
        &mut self,
        a0: Action,
        a1: Action,
        tracers: &mut [T; 2],
    ) -> Result<VersusStep, StepError> {
        if self.is_over() {
            return Err(StepError::GameOver);
        }
        for (i, a) in [a0, a1].into_iter().enumerate() {
            if legal_actions(&self.players[i].field) & (1 << a.index()) == 0 {
                return Err(StepError::IllegalAction {
                    player: i,
                    action: a,
                });
            }
        }
        let pair = self.tsumo.pair(self.turn);
        let mut res = [PlayerTurn::default(); 2];
        for (i, a) in [a0, a1].into_iter().enumerate() {
            res[i].placement =
                place_and_resolve_with(&mut self.players[i], pair, a, &mut tracers[i]);
        }
        for (i, r) in res.iter_mut().enumerate() {
            if r.placement.chain.chains == 0 {
                r.ojama_dropped = fall_ojama(&mut self.players[i]);
                if r.ojama_dropped > 0 {
                    let st = &self.players[i];
                    tracers[i].frame(FrameKind::OjamaDropped, &st.field, 0, 0, st.score);
                }
            }
        }
        let sent = exchange(
            &mut self.players,
            [res[0].placement.attack, res[1].placement.attack],
        );
        for i in 0..2 {
            res[i].ojama_sent = sent[i];
            res[i].dead = self.players[i].field.is_dead();
        }
        self.turn += 1;
        self.outcome = match (res[0].dead, res[1].dead) {
            (true, true) => Some(Outcome::Draw),
            (true, false) => Some(Outcome::Win(1)),
            (false, true) => Some(Outcome::Win(0)),
            (false, false) => None,
        };
        Ok(VersusStep {
            players: res,
            outcome: self.outcome,
        })
    }

    /// テスト・解析用: プレイヤー状態を直接書き換える。
    pub fn player_mut(&mut self, i: usize) -> &mut PlayerState {
        &mut self.players[i]
    }
}
