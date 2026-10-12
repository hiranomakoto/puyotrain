//! 2人対戦(時間のモデル)。
//!
//! 時間はフレーム(1/60秒)の整数で数える。各プレイヤーは状態(操作待ち・設置中・連鎖中・
//! おじゃま落下中)と、その状態が終わる時刻を持ち、早い出来事から順に処理する。
//! AI に手を聞くのは操作待ちのときだけ([`TimedVersus::needs_action`])。
//!
//! 1手の流れ(プレイヤーごと):
//!
//! 1. 操作待ちで手を決める([`TimedVersus::act`])。設置にかかる時間は盤面から計算する([`Timing`])。
//! 2. 着地したら、消えるぷよがあれば連鎖を1つずつ進める。連鎖1つごとに得点をおじゃまに換算し、
//!    まず自分の予告(確定 → 未確定の順)を相殺し、残りを相手の「未確定」の予告に加える。
//!    自分の連鎖が終わった時点で、相手に送った分は相手の「確定」の予告になる。
//! 3. 着地して連鎖が起きなかったとき、確定した予告おじゃまが最大30個降る
//!    (相手の連鎖が送ってくる途中の未確定分は降らない)。
//! 4. 着地(連鎖なし・おじゃまなし)、連鎖の終了、おじゃまの落下の終了のたびに窒息を判定し、
//!    操作待ちに戻る。両者が同じフレームで窒息したら引き分け。
//!
//! ツモの列は両者共通だが、手を進める速さが違えば列の別の位置にいる(各自が位置を持つ)。

use crate::action::{legal_actions, place_pair, Action, ActionMask};
use crate::chain::step_score;
use crate::color::Pair;
use crate::field::Field;
use crate::game::{ojama_seed, StepError};
use crate::phase::{ojama_pattern, PlayerState};
use crate::rules::{MAX_OJAMA_DROP, VISIBLE_HEIGHT};
use crate::score::{attack_ojama, offset};
use crate::tsumo::Tsumo;
use crate::versus::Outcome;

/// 組ぷよの出現位置(軸ぷよの段。0始まりで11 = 12段目)。設置の落下段数の基準。
pub const SPAWN_ROW: usize = 11;

/// 時間の定数(フレーム)。既定値はぷよぷよ通の目安(高速落下を使う前提)。
///
/// - 設置 = `spawn` + `drop_per_row` × 落下段数 + `landing`
///   (ちぎりがあれば + `chigiri_fall`[段差] + `landing`)
/// - 連鎖1つ = `vanish` + 落下があれば (`chain_drop_per_row` × 最大落下段数 + `chain_landing`)
/// - おじゃま = `chigiri_fall`[落下段数] + 接地(3個以下 `ojama_landing[0]`、18個以下 `[1]`、それ以上 `[2]`)
///
/// 値は Puyo Nexus の解析と puyoai の定数に基づく近似(数フレームの誤差はありうる)。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Timing {
    pub spawn: u32,
    pub drop_per_row: u32,
    pub landing: u32,
    /// ちぎり(と、おじゃま)の落下時間。添字は落下段数 − 1(1〜12段)。
    pub chigiri_fall: [u32; 12],
    pub vanish: u32,
    pub chain_drop_per_row: u32,
    pub chain_landing: u32,
    pub ojama_landing: [u32; 3],
    /// この時刻(フレーム)に達したら打ち切り。
    pub max_frames: u64,
}

impl Default for Timing {
    fn default() -> Self {
        Timing {
            spawn: 12,
            drop_per_row: 2,
            landing: 20,
            chigiri_fall: [10, 15, 19, 22, 25, 28, 31, 33, 35, 37, 39, 41],
            vanish: 50,
            chain_drop_per_row: 2,
            chain_landing: 20,
            ojama_landing: [8, 24, 32],
            max_frames: 60 * 60 * 5,
        }
    }
}

impl Timing {
    /// `rows` 段(1〜12、範囲外は端に丸める)のちぎり・おじゃまの落下時間。
    #[inline]
    pub fn fall(&self, rows: usize) -> u32 {
        self.chigiri_fall[rows.clamp(1, 12) - 1]
    }

    /// ちぎれたぷよが自由落下を始めてから `elapsed` フレームで落ちた段数(描画用。最大 `gap` 段)。
    /// `chigiri_fall` の表(n 段落ちるのにかかる時間)を逆に引き、表の間は直線で補う。
    /// 表の間隔がだんだん短くなるので、落ちるほど速くなる。
    pub fn free_fall_rows(&self, elapsed: u64, gap: usize) -> f32 {
        let gap = gap.min(12);
        let mut prev = 0u64;
        for n in 1..=gap {
            let t = self.chigiri_fall[n - 1] as u64;
            if elapsed < t {
                return (n - 1) as f32 + (elapsed - prev) as f32 / (t - prev).max(1) as f32;
            }
            prev = t;
        }
        gap as f32
    }

    /// 組ぷよを置くのにかかる時間(置く前の盤面から)。
    pub fn place_frames(&self, field: &Field, action: Action) -> u32 {
        let h = field.heights();
        let (a, c) = (action.axis_col(), action.child_col());
        let (first, gap) = if a == c {
            (h[a], 0)
        } else {
            (h[a].max(h[c]), h[a].abs_diff(h[c]))
        };
        let mut t =
            self.spawn + self.drop_per_row * SPAWN_ROW.saturating_sub(first) as u32 + self.landing;
        if gap > 0 {
            t += self.fall(gap) + self.landing;
        }
        t
    }

    /// 連鎖1つ分の時間(消えた後、落ちる前の盤面から)。
    pub fn link_frames(&self, after_clear: &Field) -> u32 {
        let fall = after_clear.max_fall_rows();
        if fall == 0 {
            self.vanish
        } else {
            self.vanish + self.chain_drop_per_row * fall + self.chain_landing
        }
    }

    /// おじゃま `n` 個が降る時間(降る前の盤面と、列ごとの個数から)。
    pub fn ojama_frames(&self, field: &Field, counts: &[u32]) -> u32 {
        let h = field.heights();
        let lowest = counts
            .iter()
            .zip(h.iter())
            .filter(|(&k, _)| k > 0)
            .map(|(_, &hc)| hc)
            .min()
            .unwrap_or(VISIBLE_HEIGHT);
        let n: u32 = counts.iter().sum();
        let landing = if n <= 3 {
            self.ojama_landing[0]
        } else if n <= 18 {
            self.ojama_landing[1]
        } else {
            self.ojama_landing[2]
        };
        self.fall(VISIBLE_HEIGHT.saturating_sub(lowest)) + landing
    }
}

/// 描画用の見た目([`TimedVersus::display`])。
#[derive(Clone, Debug, PartialEq)]
pub struct Display {
    pub field: Field,
    /// 消去の演出中に消えるセル(ビットボード)。
    pub cleared: u128,
    /// 連鎖中なら何連鎖目か(連鎖中でなければ0)。
    pub chain: u32,
    /// 落下中の組ぷよ (段(小数、0 = 1段目), 列, 色)。
    pub falling: Vec<(f32, usize, crate::color::Puyo)>,
}

/// プレイヤーの状態。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    /// 操作待ち(手を決める)。
    Ready,
    /// 設置中。`since` に手を決め、`until` に着地する。
    Placing {
        since: u64,
        until: u64,
        action: Action,
    },
    /// 連鎖中。`chain` 連鎖目を解決済みで、`until` に次の連鎖の判定をする。
    Chaining { until: u64, chain: u32 },
    /// おじゃま落下中。`until` に落下を終える。
    OjamaFalling { until: u64 },
    /// 窒息した。
    Dead,
}

/// 1プレイヤー分の状態。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimedPlayer {
    /// 盤面・確定した予告おじゃま(`pending`)・得点など。
    pub state: PlayerState,
    /// 相手の連鎖が送ってくる途中の予告おじゃま(相手の連鎖が終わると確定する)。
    pub unconfirmed: u32,
    /// 次に使うツモの位置。
    pub tsumo_index: u64,
    pub phase: Phase,
    /// 置いた手数。
    pub moves: u64,
    /// この対局の最大連鎖数。
    pub max_chain: u32,
    /// この対局で連鎖が生んだおじゃまの合計(相殺に使った分も含む)。
    pub attack_total: u64,
    /// この対局で、自分の連鎖が予告おじゃま(確定 + 未確定)を相殺した数の合計。
    pub offset_total: u64,
    /// 今の(または最後の)連鎖の開始時刻(1連鎖目を始めた時刻)。
    pub chain_start: u64,
    /// 最後に撃ち終えた連鎖の連鎖数と、その開始時刻(まだ無ければ 0)。
    pub last_chain: u32,
    pub last_chain_start: u64,
    /// 最大連鎖(`max_chain`)を撃ち始めた時刻。
    pub max_chain_start: u64,
    /// 直前の連鎖1つの開始時刻・消える前の盤面・消えたセル(描画用)。
    pub link_start: u64,
    pub link_before: Field,
    pub link_cleared: u128,
    /// この手で始まった連鎖の攻撃に全消しボーナスを乗せたか。
    bonus_used: bool,
}

impl TimedPlayer {
    fn new(seed: u64) -> Self {
        TimedPlayer {
            state: PlayerState::new(seed),
            unconfirmed: 0,
            tsumo_index: 0,
            phase: Phase::Ready,
            moves: 0,
            max_chain: 0,
            attack_total: 0,
            offset_total: 0,
            chain_start: 0,
            last_chain: 0,
            last_chain_start: 0,
            max_chain_start: 0,
            link_start: 0,
            link_before: Field::new(),
            link_cleared: 0,
            bonus_used: false,
        }
    }

    /// 予告おじゃまの合計(確定 + 未確定)。
    pub fn total_pending(&self) -> u32 {
        self.state.pending + self.unconfirmed
    }

    /// 現在の状態が終わる時刻(操作待ち・窒息なら None)。
    fn until(&self) -> Option<u64> {
        match self.phase {
            Phase::Placing { until, .. }
            | Phase::Chaining { until, .. }
            | Phase::OjamaFalling { until } => Some(until),
            Phase::Ready | Phase::Dead => None,
        }
    }
}

/// 2人対戦(時間のモデル)。`Copy` なので clone が安い。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimedVersus {
    players: [TimedPlayer; 2],
    tsumo: Tsumo,
    timing: Timing,
    now: u64,
    outcome: Option<Outcome>,
    timed_out: bool,
}

impl TimedVersus {
    pub fn new(seed: u64) -> Self {
        Self::with_timing(seed, Timing::default())
    }

    pub fn with_timing(seed: u64, timing: Timing) -> Self {
        TimedVersus {
            players: [
                TimedPlayer::new(ojama_seed(seed, 0)),
                TimedPlayer::new(ojama_seed(seed, 1)),
            ],
            tsumo: Tsumo::new(seed),
            timing,
            now: 0,
            outcome: None,
            timed_out: false,
        }
    }

    pub fn player(&self, i: usize) -> &TimedPlayer {
        &self.players[i]
    }

    pub fn timing(&self) -> &Timing {
        &self.timing
    }

    /// 現在の時刻(フレーム)。
    pub fn now(&self) -> u64 {
        self.now
    }

    pub fn seed(&self) -> u64 {
        self.tsumo.seed()
    }

    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    /// 打ち切り(`Timing::max_frames`)で終わったか。
    pub fn timed_out(&self) -> bool {
        self.timed_out
    }

    pub fn is_over(&self) -> bool {
        self.outcome.is_some() || self.timed_out
    }

    /// プレイヤー `i` の現在(k=0)・ネクスト(k=1)・ネクネク(k=2)。
    pub fn pair(&self, i: usize, k: u64) -> Pair {
        self.tsumo.pair(self.players[i].tsumo_index + k)
    }

    pub fn visible_pairs(&self, i: usize) -> [Pair; 3] {
        [self.pair(i, 0), self.pair(i, 1), self.pair(i, 2)]
    }

    /// 手を決める必要があるプレイヤー(操作待ち)。終了後は両方 false。
    pub fn needs_action(&self) -> [bool; 2] {
        if self.is_over() {
            return [false; 2];
        }
        [0, 1].map(|i| self.players[i].phase == Phase::Ready)
    }

    /// プレイヤー `i` の合法手(操作待ちでなければ0)。
    pub fn legal_actions(&self, i: usize) -> ActionMask {
        if self.needs_action()[i] {
            legal_actions(&self.players[i].state.field)
        } else {
            0
        }
    }

    /// プレイヤー `i` のルールベース(2手読み)の手。操作待ちでなければ `None`。
    pub fn greedy_action(&self, i: usize, rng: &mut crate::tsumo::Rng) -> Option<Action> {
        if !self.needs_action()[i] {
            return None;
        }
        crate::bot::greedy_action(
            &self.players[i].state.field,
            [self.pair(i, 0), self.pair(i, 1)],
            rng,
        )
    }

    /// 操作待ちのプレイヤー `i` が手を決める(設置を始める)。時刻は進めない。
    pub fn act(&mut self, i: usize, action: Action) -> Result<(), StepError> {
        if self.is_over() {
            return Err(StepError::GameOver);
        }
        let p = &mut self.players[i];
        if p.phase != Phase::Ready || legal_actions(&p.state.field) & (1 << action.index()) == 0 {
            return Err(StepError::IllegalAction { player: i, action });
        }
        let until = self.now + self.timing.place_frames(&p.state.field, action) as u64;
        p.phase = Phase::Placing {
            since: self.now,
            until,
            action,
        };
        Ok(())
    }

    /// 誰かが操作待ちになるか、対局が終わるまで時刻を進める。
    pub fn advance(&mut self) {
        self.advance_until(u64::MAX);
    }

    /// `advance` と同じだが、時刻 `limit` を超える出来事は処理しない(描画用)。
    /// 誰も操作待ちでなく、次の出来事が `limit` より後なら、時刻を `limit` にして止まる。
    pub fn advance_until(&mut self, limit: u64) {
        while !self.is_over() && !self.needs_action().iter().any(|&b| b) {
            let Some(t) = self.players.iter().filter_map(TimedPlayer::until).min() else {
                break;
            };
            if t >= self.timing.max_frames && limit >= self.timing.max_frames {
                self.now = self.timing.max_frames;
                self.timed_out = true;
                break;
            }
            if t > limit {
                self.now = self.now.max(limit);
                break;
            }
            self.now = t;
            for i in 0..2 {
                if self.players[i].until() == Some(t) {
                    self.handle_event(i);
                }
            }
            let dead = [0, 1].map(|i| self.players[i].phase == Phase::Dead);
            self.outcome = match dead {
                [true, true] => Some(Outcome::Draw),
                [true, false] => Some(Outcome::Win(1)),
                [false, true] => Some(Outcome::Win(0)),
                [false, false] => None,
            };
        }
    }

    /// プレイヤー `i` の現在の状態が終わったときの処理。
    fn handle_event(&mut self, i: usize) {
        match self.players[i].phase {
            Phase::Placing { action, .. } => {
                let pair = self.pair(i, 0);
                let p = &mut self.players[i];
                place_pair(&mut p.state.field, pair, action);
                p.tsumo_index += 1;
                p.moves += 1;
                p.bonus_used = false;
                if !self.start_link(i, 1) {
                    self.after_quiet_landing(i);
                }
            }
            Phase::Chaining { chain, .. } => {
                if !self.start_link(i, chain + 1) {
                    self.end_chain(i, chain);
                }
            }
            Phase::OjamaFalling { .. } => self.settle(i),
            Phase::Ready | Phase::Dead => {}
        }
    }

    /// `chain` 連鎖目を始める(消えるぷよが無ければ false)。消去・落下・攻撃をここで行い、
    /// 連鎖1つ分の時間だけ連鎖中にする。
    fn start_link(&mut self, i: usize, chain: u32) -> bool {
        let timing = self.timing;
        let now = self.now;
        let p = &mut self.players[i];
        let before = p.state.field;
        let Some((clear, cleared)) = p.state.field.clear_groups_masked() else {
            return false;
        };
        p.link_start = now;
        p.link_before = before;
        p.link_cleared = cleared;
        let frames = timing.link_frames(&p.state.field);
        p.state.field.apply_gravity();
        let score = step_score(chain, &clear);
        p.state.score += score as u64;
        let bonus = p.state.all_clear && !p.bonus_used;
        if bonus {
            p.bonus_used = true;
        }
        let mut attack = attack_ojama(score, &mut p.state.carry, bonus);
        p.attack_total += attack as u64;
        // 自分の予告を相殺(確定分 → 未確定分)し、残りを相手の未確定の予告へ
        let generated = attack;
        attack = offset(attack, &mut p.state.pending);
        attack = offset(attack, &mut p.unconfirmed);
        p.offset_total += (generated - attack) as u64;
        if chain == 1 {
            p.chain_start = now;
        }
        if chain > p.max_chain {
            p.max_chain = chain;
            p.max_chain_start = p.chain_start;
        }
        p.phase = Phase::Chaining {
            until: now + frames as u64,
            chain,
        };
        self.players[1 - i].unconfirmed += attack;
        true
    }

    /// `chain` 連鎖で連鎖が終わった。相手に送った分を確定させ、全消しを判定して操作待ちに戻る。
    fn end_chain(&mut self, i: usize, chain: u32) {
        let opp = &mut self.players[1 - i];
        opp.state.pending += opp.unconfirmed;
        opp.unconfirmed = 0;
        let p = &mut self.players[i];
        p.last_chain = chain;
        p.last_chain_start = p.chain_start;
        if p.bonus_used {
            p.state.all_clear = false;
        }
        if p.state.field.is_empty() {
            p.state.all_clear = true;
        }
        self.settle(i);
    }

    /// 連鎖せずに着地した。確定した予告おじゃまがあれば降らせる。
    fn after_quiet_landing(&mut self, i: usize) {
        let timing = self.timing;
        let now = self.now;
        let st = &mut self.players[i].state;
        let n = st.pending.min(MAX_OJAMA_DROP);
        if n == 0 {
            self.settle(i);
            return;
        }
        let counts = ojama_pattern(n, &mut st.ojama_rng);
        let frames = timing.ojama_frames(&st.field, &counts);
        st.field.drop_ojama_columns(&counts);
        st.pending -= n;
        self.players[i].phase = Phase::OjamaFalling {
            until: now + frames as u64,
        };
    }

    /// 一連の処理が終わった。窒息を判定し、操作待ちに戻る。
    fn settle(&mut self, i: usize) {
        let p = &mut self.players[i];
        p.phase = if p.state.field.is_dead() {
            Phase::Dead
        } else {
            Phase::Ready
        };
    }

    /// 描画用: プレイヤー `i` の今の見た目。
    ///
    /// - 連鎖1つの消去の演出中(開始から `vanish` フレームの間)は、消える前の盤面と消えるセル。
    /// - 設置中は、落下中の組ぷよ2個の (段(小数), 列, 色)。設置時間の式と同じ区切りで動かす:
    ///   出現位置で止まる(`spawn`)→ 2個一緒に一定の速さで落ちる(`drop_per_row` × 段数)→
    ///   先に着地した側が接地する(`landing`)→ ちぎりなら低い側だけが自由落下する
    ///   (`chigiri_fall` の表に沿って加速。[`Timing::free_fall_rows`])→ 接地。
    pub fn display(&self, i: usize) -> Display {
        let p = &self.players[i];
        let mut d = Display {
            field: p.state.field,
            cleared: 0,
            chain: 0,
            falling: Vec::new(),
        };
        match p.phase {
            Phase::Chaining { chain, .. } => {
                d.chain = chain;
                if self.now < p.link_start + self.timing.vanish as u64 {
                    d.field = p.link_before;
                    d.cleared = p.link_cleared;
                }
            }
            Phase::Placing { since, action, .. } => {
                let pair = self.pair(i, 0);
                let h = p.state.field.heights();
                let (a, c) = (action.axis_col(), action.child_col());
                let t = &self.timing;
                // 設置時間の式と同じ区切り: 出現 → 操作による落下(2個一緒) → 接地 →
                // (ちぎりなら)低い側だけ自由落下 → 接地
                let first = if a == c { h[a] } else { h[a].max(h[c]) };
                let drop_rows = SPAWN_ROW.saturating_sub(first);
                let elapsed = self.now.saturating_sub(since + t.spawn as u64);
                let fallen = (elapsed as f32 / t.drop_per_row.max(1) as f32).min(drop_rows as f32);
                let base = SPAWN_ROW as f32 - fallen; // 2個一緒に落ちている間の下側の段
                                                      // ちぎれて自由落下する時間(先に着地した側の接地が終わってから)
                let free = elapsed
                    .saturating_sub((t.drop_per_row as usize * drop_rows) as u64)
                    .saturating_sub(t.landing as u64);
                d.falling = match action.rotation() {
                    crate::action::Rotation::Up => {
                        vec![(base, a, pair.axis), (base + 1.0, a, pair.child)]
                    }
                    crate::action::Rotation::Down => {
                        vec![(base + 1.0, a, pair.axis), (base, a, pair.child)]
                    }
                    _ => [(a, pair.axis), (c, pair.child)]
                        .iter()
                        .map(|&(col, color)| {
                            let gap = first - h[col]; // 先に着地する側は0
                            let row = if gap == 0 {
                                base
                            } else {
                                base - t.free_fall_rows(free, gap)
                            };
                            (row, col, color)
                        })
                        .collect(),
                };
            }
            _ => {}
        }
        d
    }

    /// テスト・解析用: プレイヤー状態を直接書き換える。
    pub fn player_mut(&mut self, i: usize) -> &mut TimedPlayer {
        &mut self.players[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Rotation;
    use crate::color::Puyo;

    fn up(col: usize) -> Action {
        Action::from_placement(col, Rotation::Up).unwrap()
    }

    fn right(col: usize) -> Action {
        Action::from_placement(col, Rotation::Right).unwrap()
    }

    #[test]
    fn place_frames_follow_the_formula() {
        let t = Timing::default();
        let empty = Field::new();
        // 空の列に縦置き: 12 + 2 × 11 + 20
        assert_eq!(t.place_frames(&empty, up(0)), 54);
        // 高さ4の列: 12 + 2 × 7 + 20
        let f = Field::from_rows(&["R.....", "G.....", "B.....", "Y....."]).unwrap();
        assert_eq!(t.place_frames(&f, up(0)), 46);
        // 横置きでちぎり(高さ4と0): 12 + 2 × 7 + 20 + ちぎり4段(22) + 20
        assert_eq!(t.place_frames(&f, right(0)), 46 + 22 + 20);
        // 高さが同じ横置きはちぎりなし
        assert_eq!(t.place_frames(&empty, right(2)), 54);
    }

    #[test]
    fn link_frames_depend_on_fall() {
        let t = Timing::default();
        // 落下なし: 50
        assert_eq!(t.link_frames(&Field::new()), 50);
        // 3段浮いている: 50 + 2 × 3 + 20
        let f = Field::from_rows(&["R.....", "......", "......", "......"]).unwrap();
        assert_eq!(f.max_fall_rows(), 3);
        assert_eq!(t.link_frames(&f), 76);
    }

    #[test]
    fn ojama_frames_depend_on_count_and_height() {
        let t = Timing::default();
        let empty = Field::new();
        // 12段落ちて(41)、6個なので接地 24
        assert_eq!(t.ojama_frames(&empty, &[1, 1, 1, 1, 1, 1]), 41 + 24);
        // 2個なら接地 8
        assert_eq!(t.ojama_frames(&empty, &[1, 1, 0, 0, 0, 0]), 41 + 8);
        // 30個は接地 32
        assert_eq!(t.ojama_frames(&empty, &[5; 6]), 41 + 32);
    }

    /// 両者が手を出し続けて最後まで打つ(合法手の先頭を選ぶ)。
    fn play_out(g: &mut TimedVersus) {
        let mut guard = 0;
        while !g.is_over() {
            for i in 0..2 {
                if g.needs_action()[i] {
                    let m = g.legal_actions(i);
                    g.act(i, Action::new(m.trailing_zeros() as usize).unwrap())
                        .unwrap();
                }
            }
            g.advance();
            guard += 1;
            assert!(guard < 10_000);
        }
    }

    #[test]
    fn game_ends_and_time_moves_forward() {
        let mut g = TimedVersus::new(1);
        assert_eq!(g.needs_action(), [true, true]);
        play_out(&mut g);
        assert!(g.outcome().is_some() || g.timed_out());
        assert!(g.now() > 0);
    }

    #[test]
    fn players_move_at_their_own_pace() {
        let mut g = TimedVersus::new(3);
        // P0 は空の列(54F)、P1 は横置き(54F)…同じ時間なら同時に操作待ちになる
        g.act(0, up(0)).unwrap();
        g.act(1, up(5)).unwrap();
        g.advance();
        assert_eq!(g.now(), 54);
        assert_eq!(g.needs_action(), [true, true]);
        assert_eq!(g.player(0).tsumo_index, 1);
        // P0 だけ置く。P1 は操作待ちのままなので時刻は進まない
        g.act(0, up(0)).unwrap();
        g.advance();
        assert_eq!(g.now(), 54);
        assert_eq!(g.needs_action(), [false, true]);
    }

    #[test]
    fn chain_attack_is_unconfirmed_until_the_chain_ends() {
        let mut g = TimedVersus::new(5);
        // P0 の列0に「今のツモの軸の色」を3個置いておく。縦置き(軸が下)で4個つながって消える
        let c = g.pair(0, 0).axis.to_char();
        let row = format!("{c}.....");
        g.player_mut(0).state.field = Field::from_rows(&[&row, &row, &row]).unwrap();
        g.player_mut(0).state.carry = 69; // 1連鎖(40点)+ 端数69 = 109点 → おじゃま1個
        g.act(0, up(0)).unwrap();
        g.act(1, up(5)).unwrap();
        g.advance();
        // P0 は高さ3の列なので先に着地(48F)して連鎖中。P1 が着地した(54F)時点で止まる
        assert_eq!(g.now(), 54);
        assert!(matches!(
            g.player(0).phase,
            Phase::Chaining { chain: 1, .. }
        ));
        assert_eq!(g.player(0).max_chain, 1);
        assert_eq!(g.player(1).unconfirmed, 1);
        assert_eq!(g.player(1).state.pending, 0);
        // P0 の連鎖中に P1 が消さずに置いても、未確定の1個は降らない
        while matches!(g.player(0).phase, Phase::Chaining { .. }) {
            if g.needs_action()[1] {
                assert_eq!(g.player(1).unconfirmed, 1);
                g.act(1, up(5)).unwrap();
            }
            g.advance();
        }
        // P0 の連鎖が終わると確定する
        assert_eq!(g.player(1).unconfirmed, 0);
        assert_eq!(g.player(1).total_pending(), 1);
        assert_eq!(g.player(1).state.field.plane(Puyo::Ojama), 0);
    }

    #[test]
    fn display_shows_vanishing_puyos_then_the_fallen_board() {
        let mut g = TimedVersus::new(5);
        let c = g.pair(0, 0).axis.to_char();
        let row = format!("{c}.....");
        g.player_mut(0).state.field = Field::from_rows(&[&row, &row, &row]).unwrap();
        g.act(0, up(0)).unwrap();
        g.act(1, up(5)).unwrap();
        // 高さ3の列: 12 + 2 × 8 + 20 = 48F で着地して1連鎖目が始まる
        g.advance_until(48);
        assert_eq!(g.now(), 48);
        let d = g.display(0);
        assert_eq!(d.chain, 1);
        assert!(d.cleared.count_ones() >= 4);
        // 消える前の盤面
        assert!(d.field.count() >= 4);
        // 消去の演出(50F)が終わると、消えた後の盤面になる(途中で操作待ちになる P1 は置き続ける)
        while g.now() < 48 + 50 {
            if g.needs_action()[1] {
                g.act(1, up(5)).unwrap();
            }
            g.advance_until(48 + 50);
        }
        let d = g.display(0);
        assert_eq!(d.cleared, 0);
        assert_eq!(d.field, g.player(0).state.field);
    }

    /// P0 の表示を時刻 `t` で取る(途中で操作待ちになる P1 は列5に置き続ける)。
    fn display_at(g: &mut TimedVersus, t: u64) -> Display {
        while g.now() < t {
            if g.needs_action()[1] {
                g.act(1, up(5)).unwrap();
            }
            g.advance_until(t);
        }
        g.display(0)
    }

    #[test]
    fn chigiri_drops_together_then_the_lower_side_falls_alone() {
        let mut g = TimedVersus::new(13);
        // 列0の高さ4、列1の高さ0。右向きの横置きは列0(軸)が先に着地し、子が4段ちぎれる
        g.player_mut(0).state.field =
            Field::from_rows(&["R.....", "G.....", "B.....", "Y....."]).unwrap();
        g.act(0, right(0)).unwrap();
        let rows = |d: &Display| (d.falling[0].0, d.falling[1].0);
        // 出現中(12F)は出現位置
        assert_eq!(rows(&display_at(&mut g, 5)), (11.0, 11.0));
        // 操作による落下: 2個一緒に 2F/段(18F で3段)
        assert_eq!(rows(&display_at(&mut g, 18)), (8.0, 8.0));
        // 26F で軸が着地(段4)。接地の20F の間、子はまだくっついている
        assert_eq!(rows(&display_at(&mut g, 26)), (4.0, 4.0));
        assert_eq!(rows(&display_at(&mut g, 45)), (4.0, 4.0));
        // 46F から子だけが自由落下: 10F で1段、22F で4段(表どおり加速)
        assert_eq!(rows(&display_at(&mut g, 56)), (4.0, 3.0));
        let d = display_at(&mut g, 58); // 12F 後: 1段(10F)と2段(15F)の間
        assert!(d.falling[1].0 < 3.0 && d.falling[1].0 > 2.0);
        assert_eq!(rows(&display_at(&mut g, 68)), (4.0, 0.0));
        // 子の接地(20F)が終わる 88F に着地する(設置時間 = 46 + 22 + 20)
        assert_eq!(display_at(&mut g, 87).falling.len(), 2);
        assert!(display_at(&mut g, 88).falling.is_empty());
    }

    #[test]
    fn vertical_drop_moves_both_together() {
        let mut g = TimedVersus::new(13);
        g.act(0, up(0)).unwrap();
        // 空の列: 軸が下(段11 → 0)、子が上(段12 → 1)。12F 後から 2F/段
        let d = display_at(&mut g, 12 + 2 * 5);
        assert_eq!((d.falling[0].0, d.falling[1].0), (6.0, 7.0));
        let d = display_at(&mut g, 12 + 22);
        assert_eq!((d.falling[0].0, d.falling[1].0), (0.0, 1.0));
    }

    #[test]
    fn free_fall_follows_the_table() {
        let t = Timing::default();
        assert_eq!(t.free_fall_rows(0, 3), 0.0);
        assert_eq!(t.free_fall_rows(10, 3), 1.0);
        assert_eq!(t.free_fall_rows(15, 3), 2.0);
        assert_eq!(t.free_fall_rows(17, 3), 2.5);
        assert_eq!(t.free_fall_rows(100, 3), 3.0);
    }

    #[test]
    fn records_last_chain_and_offsets() {
        let mut g = TimedVersus::new(5);
        let c = g.pair(0, 0).axis.to_char();
        let row = format!("{c}.....");
        g.player_mut(0).state.field = Field::from_rows(&[&row, &row, &row]).unwrap();
        g.player_mut(0).state.carry = 69; // 1連鎖で1個以上
        g.player_mut(0).state.pending = 6;
        g.act(0, up(0)).unwrap();
        g.act(1, up(5)).unwrap();
        // P0 の連鎖が終わるまで進める
        while g.player(0).last_chain == 0 {
            if g.needs_action()[1] {
                g.act(1, up(5)).unwrap();
            }
            g.advance();
        }
        let p = g.player(0);
        assert_eq!(p.last_chain, 1);
        assert_eq!(p.last_chain_start, 48); // 高さ3の列に置いて 48F で着地・発火
        assert_eq!(p.max_chain_start, 48);
        assert!(p.offset_total >= 1);
        assert_eq!(p.offset_total + p.state.pending as u64, 6);
    }

    #[test]
    fn offset_and_confirmation() {
        let mut g = TimedVersus::new(7);
        // P1 に未確定の予告が10個(P0 の連鎖中という想定)、確定が0
        g.player_mut(0).phase = Phase::Chaining {
            until: 100,
            chain: 1,
        };
        g.player_mut(1).unconfirmed = 10;
        // P1 は操作待ちで、消えない手を置く(54F で着地)→ 未確定分は降らない
        g.act(1, up(5)).unwrap();
        g.advance();
        assert_eq!(g.now(), 54);
        assert_eq!(g.player(1).state.field.count(), 2);
        assert_eq!(g.player(1).unconfirmed, 10);
        // 次の手の途中(時刻100)で P0 の連鎖が終わる → P1 の予告が確定する
        g.act(1, up(5)).unwrap();
        g.advance();
        assert_eq!(g.now(), 100);
        assert_eq!(g.player(1).unconfirmed, 0);
        assert_eq!(g.player(1).state.pending, 10);
        // P1 の着地(54 + 50 = 104F)で確定した10個が降る。降り終わるまで P1 は操作待ちにならない
        g.act(0, up(0)).unwrap();
        g.advance();
        assert!(g.now() > 104);
        assert_eq!(g.player(1).state.field.plane(Puyo::Ojama).count_ones(), 10);
        assert_eq!(g.player(1).total_pending(), 0);
    }

    #[test]
    fn confirmed_pending_drops_on_quiet_landing() {
        let mut g = TimedVersus::new(9);
        g.player_mut(0).state.pending = 12;
        g.act(0, up(0)).unwrap();
        g.act(1, up(0)).unwrap();
        g.advance();
        // 着地(54F)で P0 は連鎖しないので12個(2段)降り始める
        assert_eq!(g.now(), 54);
        assert!(matches!(g.player(0).phase, Phase::OjamaFalling { .. }));
        assert_eq!(g.player(0).state.pending, 0);
        assert_eq!(g.player(0).state.field.count(), 2 + 12);
        assert_eq!(g.needs_action(), [false, true]);
    }

    #[test]
    fn deterministic() {
        let mut a = TimedVersus::new(11);
        let mut b = TimedVersus::new(11);
        play_out(&mut a);
        play_out(&mut b);
        assert_eq!(a, b);
    }
}
