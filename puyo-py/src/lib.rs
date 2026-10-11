//! puyo-core の Python バインディング(モジュール名 `puyotrain._puyo`)。

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use puyo_core::afterstate::{
    action_features, versus_action_features, versus_info, ACTION_FEATURE_NAMES,
    NUM_ACTION_FEATURES, NUM_VERSUS_ACTION_FEATURES, NUM_VERSUS_INFO, VERSUS_ACTION_FEATURE_NAMES,
    VERSUS_INFO_NAMES,
};
use puyo_core::bot::safe_actions;
use puyo_core::field::FULL_MASK;
use puyo_core::potential::potential_chain;
use puyo_core::rules::{HEIGHT, NUM_ACTIONS, NUM_COLORS, WIDTH};
use puyo_core::timed::{Phase, TimedVersus, Timing};
use puyo_core::timed_features::{
    NUM_TIMED_ACTION_FEATURES, NUM_TIMED_INFO, TIMED_ACTION_FEATURE_NAMES, TIMED_INFO_NAMES,
};

mod timed_batch;
use puyo_core::{
    Action, ActionMask, Field, Game, Outcome, Pair, PlayerState, Puyo, Rng, StepError, TraceFrame,
    Versus, VersusStep,
};

/// 盤面平面の数: 空 + 4色 + おじゃま。
const NUM_BOARD_PLANES: usize = NUM_COLORS + 2;
const BOARD_LEN: usize = NUM_BOARD_PLANES * HEIGHT * WIDTH;
/// 公開される組ぷよの数(現在・ネクスト・ネクネク)。
const NUM_VISIBLE_PAIRS: usize = 3;

fn to_py_err(e: StepError) -> PyErr {
    match e {
        StepError::IllegalAction { .. } => PyValueError::new_err(e.to_string()),
        StepError::GameOver => PyRuntimeError::new_err(e.to_string()),
    }
}

fn action(index: usize) -> PyResult<Action> {
    Action::new(index)
        .ok_or_else(|| PyValueError::new_err(format!("action must be in 0..{}", NUM_ACTIONS)))
}

/// 盤面を (平面, 段, 列) のワンホットで書き込む。平面0 = 空、1..=4 = 色、5 = おじゃま。段0 = 1段目。
fn write_board(out: &mut [u8], st: &PlayerState) {
    let plane_len = HEIGHT * WIDTH;
    let mut put = |plane: usize, bits: u128| {
        let mut b = bits & FULL_MASK;
        while b != 0 {
            let i = b.trailing_zeros() as usize;
            b &= b - 1;
            let (col, row) = (i / 16, i % 16);
            out[plane * plane_len + row * WIDTH + col] = 1;
        }
    };
    put(0, !st.field.occupied());
    for p in Puyo::COLORS.into_iter().chain([Puyo::Ojama]) {
        put(1 + p.index(), st.field.plane(p));
    }
}

fn mask_vec(mask: ActionMask) -> Vec<i8> {
    (0..NUM_ACTIONS).map(|i| ((mask >> i) & 1) as i8).collect()
}

fn pieces_vec(pairs: &[Pair; NUM_VISIBLE_PAIRS]) -> Vec<u8> {
    let mut v = vec![0u8; NUM_VISIBLE_PAIRS * 2 * NUM_COLORS];
    for (k, p) in pairs.iter().enumerate() {
        v[(k * 2) * NUM_COLORS + p.axis.index()] = 1;
        v[(k * 2 + 1) * NUM_COLORS + p.child.index()] = 1;
    }
    v
}

/// 観測 dict を作る。`opp` が無ければ相手盤面はすべて0。
fn observation<'py>(
    py: Python<'py>,
    me: &PlayerState,
    opp: Option<&PlayerState>,
    pairs: &[Pair; NUM_VISIBLE_PAIRS],
    mask: ActionMask,
) -> PyResult<Bound<'py, PyDict>> {
    let mut board = vec![0u8; 2 * BOARD_LEN];
    write_board(&mut board[..BOARD_LEN], me);
    if let Some(o) = opp {
        write_board(&mut board[BOARD_LEN..], o);
    }
    let ojama = vec![me.pending as i32, opp.map_or(0, |o| o.pending as i32)];
    let d = PyDict::new(py);
    d.set_item(
        "board",
        PyArray1::from_vec(py, board).reshape([2, NUM_BOARD_PLANES, HEIGHT, WIDTH])?,
    )?;
    d.set_item(
        "pieces",
        PyArray1::from_vec(py, pieces_vec(pairs)).reshape([NUM_VISIBLE_PAIRS, 2, NUM_COLORS])?,
    )?;
    d.set_item("ojama", PyArray1::from_vec(py, ojama))?;
    d.set_item("action_mask", PyArray1::from_vec(py, mask_vec(mask)))?;
    Ok(d)
}

/// 盤面を (段, 列) の int8 配列にする。-1 = 空、0..=3 = 色、4 = おじゃま。段0 = 1段目。
fn grid<'py>(py: Python<'py>, field: &Field) -> PyResult<Bound<'py, numpy::PyArray2<i8>>> {
    let mut v = vec![-1i8; HEIGHT * WIDTH];
    for p in Puyo::COLORS.into_iter().chain([Puyo::Ojama]) {
        let mut b = field.plane(p) & FULL_MASK;
        while b != 0 {
            let i = b.trailing_zeros() as usize;
            b &= b - 1;
            v[(i % 16) * WIDTH + i / 16] = p.index() as i8;
        }
    }
    PyArray1::from_vec(py, v).reshape([HEIGHT, WIDTH])
}

/// ビットマスクを (段, 列) の bool 配列にする。
fn mask_grid<'py>(py: Python<'py>, mask: u128) -> PyResult<Bound<'py, numpy::PyArray2<bool>>> {
    let mut v = vec![false; HEIGHT * WIDTH];
    let mut b = mask & FULL_MASK;
    while b != 0 {
        let i = b.trailing_zeros() as usize;
        b &= b - 1;
        v[(i % 16) * WIDTH + i / 16] = true;
    }
    PyArray1::from_vec(py, v).reshape([HEIGHT, WIDTH])
}

/// トレースを dict のリストにする: kind, board, cleared, chain, score。
fn frames_list<'py>(py: Python<'py>, frames: &[TraceFrame]) -> PyResult<Vec<Bound<'py, PyDict>>> {
    frames
        .iter()
        .map(|f| {
            let d = PyDict::new(py);
            d.set_item("kind", f.kind.name())?;
            d.set_item("board", grid(py, &f.field)?)?;
            d.set_item("cleared", mask_grid(py, f.cleared)?)?;
            d.set_item("chain", f.chain)?;
            d.set_item("score", f.score)?;
            Ok(d)
        })
        .collect()
}

fn versus_results<'py>(py: Python<'py>, r: &VersusStep) -> PyResult<Vec<Bound<'py, PyDict>>> {
    r.players
        .iter()
        .map(|t| {
            let d = PyDict::new(py);
            d.set_item("score", t.placement.chain.score)?;
            d.set_item("chains", t.placement.chain.chains)?;
            d.set_item("all_clear", t.placement.all_clear)?;
            d.set_item("attack", t.placement.attack)?;
            d.set_item("ojama_sent", t.ojama_sent)?;
            d.set_item("ojama_dropped", t.ojama_dropped)?;
            d.set_item("dead", t.dead)?;
            Ok(d)
        })
        .collect()
}

fn pair_tuple(p: Pair) -> (usize, usize) {
    (p.axis.index(), p.child.index())
}

/// 1人用ゲーム。
#[pyclass(name = "Game", module = "puyotrain._puyo", skip_from_py_object)]
#[derive(Clone)]
struct PyGame {
    inner: Game,
}

#[pymethods]
impl PyGame {
    #[new]
    fn new(seed: u64) -> Self {
        PyGame {
            inner: Game::new(seed),
        }
    }

    /// 1手進める。戻り値: (得点, 連鎖数, 全消し, 窒息)。
    fn step(&mut self, action_index: usize) -> PyResult<(u32, u32, bool, bool)> {
        let r = self.inner.step(action(action_index)?).map_err(to_py_err)?;
        let p = r.placement;
        Ok((p.chain.score, p.chain.chains, p.all_clear, r.dead))
    }

    /// `step` と同じ。加えて中間盤面のリスト(設置後、各連鎖の消去前・消去後・落下後)を返す。
    #[allow(clippy::type_complexity)]
    fn step_traced<'py>(
        &mut self,
        py: Python<'py>,
        action_index: usize,
    ) -> PyResult<((u32, u32, bool, bool), Vec<Bound<'py, PyDict>>)> {
        let (r, frames) = self
            .inner
            .step_traced(action(action_index)?)
            .map_err(to_py_err)?;
        let p = r.placement;
        Ok((
            (p.chain.score, p.chain.chains, p.all_clear, r.dead),
            frames_list(py, &frames)?,
        ))
    }

    /// ルールベース(2手読み)の手。同点時の乱数は `seed` から決まる。終了後は None。
    fn greedy_action(&self, seed: u64) -> Option<usize> {
        self.inner
            .greedy_action(&mut Rng::new(seed))
            .map(Action::index)
    }

    /// 自滅しない手のビットマスク(int)。置いた直後に窒息する手と、ネクストを
    /// どう置いても窒息する手を除く。すべて除かれる場合は合法手と同じ。終了後は0。
    fn safe_actions(&self) -> u32 {
        self.inner.safe_actions()
    }

    /// `safe_actions` を長さ22の int8 配列で返す。
    fn safe_action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        PyArray1::from_vec(py, mask_vec(self.inner.safe_actions()))
    }

    /// 手ごとの結果の特徴量 (22, NUM_ACTION_FEATURES) float32。列の意味は
    /// ACTION_FEATURE_NAMES。合法でない手と終了後は0。
    fn action_features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        let flat: Vec<f32> = self
            .inner
            .action_features()
            .iter()
            .flatten()
            .copied()
            .collect();
        PyArray1::from_vec(py, flat).reshape([NUM_ACTIONS, NUM_ACTION_FEATURES])
    }

    /// 潜在連鎖 (連鎖数, 得点)。どこかの列に同じ色を1〜2個足したときに起きる最大の連鎖。
    fn potential_chain(&self) -> (u32, u32) {
        let p = self.inner.potential_chain();
        (p.chains, p.score)
    }

    /// 現在の盤面 (13, 6) int8。-1 = 空、0..=3 = 色、4 = おじゃま。段0 = 1段目。
    fn board<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<i8>>> {
        grid(py, self.inner.field())
    }

    /// 合法手のビットマスク(int)。
    fn legal_actions(&self) -> u32 {
        self.inner.legal_actions()
    }

    /// 合法手マスク(長さ22の int8 配列)。
    fn action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        PyArray1::from_vec(py, mask_vec(self.inner.legal_actions()))
    }

    fn observation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        observation(
            py,
            self.inner.player(),
            None,
            &self.inner.visible_pairs(),
            self.inner.legal_actions(),
        )
    }

    /// 現在・ネクスト・ネクネクの組ぷよ [(軸, 子), ...]。色番号は 0=赤 1=緑 2=青 3=黄。
    fn visible_pairs(&self) -> Vec<(usize, usize)> {
        self.inner
            .visible_pairs()
            .into_iter()
            .map(pair_tuple)
            .collect()
    }

    #[getter]
    fn score(&self) -> u64 {
        self.inner.score()
    }

    #[getter]
    fn turn(&self) -> u64 {
        self.inner.turn()
    }

    #[getter]
    fn is_over(&self) -> bool {
        self.inner.is_over()
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    fn field_str(&self) -> String {
        self.inner.field().to_string()
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    fn __copy__(&self) -> Self {
        Clone::clone(self)
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        Clone::clone(self)
    }

    fn __repr__(&self) -> String {
        format!(
            "Game(seed={}, turn={}, score={})",
            self.inner.seed(),
            self.inner.turn(),
            self.inner.score()
        )
    }
}

/// 2人対戦(同期ターン制)。
#[pyclass(name = "Versus", module = "puyotrain._puyo", skip_from_py_object)]
#[derive(Clone)]
struct PyVersus {
    inner: Versus,
}

#[pymethods]
impl PyVersus {
    #[new]
    fn new(seed: u64) -> Self {
        PyVersus {
            inner: Versus::new(seed),
        }
    }

    /// 両者の行動で1ターン進める。各プレイヤーの結果を dict のリストで返す。
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        a0: usize,
        a1: usize,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let r = self
            .inner
            .step(action(a0)?, action(a1)?)
            .map_err(to_py_err)?;
        versus_results(py, &r)
    }

    /// `step` と同じ。加えて各プレイヤーの中間盤面のリスト [P0, P1] を返す。
    #[allow(clippy::type_complexity)]
    fn step_traced<'py>(
        &mut self,
        py: Python<'py>,
        a0: usize,
        a1: usize,
    ) -> PyResult<(Vec<Bound<'py, PyDict>>, Vec<Vec<Bound<'py, PyDict>>>)> {
        let (r, frames) = self
            .inner
            .step_traced(action(a0)?, action(a1)?)
            .map_err(to_py_err)?;
        Ok((
            versus_results(py, &r)?,
            vec![frames_list(py, &frames[0])?, frames_list(py, &frames[1])?],
        ))
    }

    /// プレイヤー `player` のルールベース(2手読み)の手。終了後は None。
    fn greedy_action(&self, player: usize, seed: u64) -> PyResult<Option<usize>> {
        check_player(player)?;
        Ok(self
            .inner
            .greedy_action(player, &mut Rng::new(seed))
            .map(Action::index))
    }

    fn board<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray2<i8>>> {
        check_player(player)?;
        grid(py, &self.inner.player(player).field)
    }

    fn legal_actions(&self, player: usize) -> PyResult<u32> {
        check_player(player)?;
        Ok(self.inner.legal_actions(player))
    }

    fn action_mask<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, PyArray1<i8>>> {
        check_player(player)?;
        Ok(PyArray1::from_vec(
            py,
            mask_vec(self.inner.legal_actions(player)),
        ))
    }

    /// プレイヤー `player` 視点の観測(自分が先)。
    fn observation<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyDict>> {
        check_player(player)?;
        let ps = self.inner.players();
        observation(
            py,
            &ps[player],
            Some(&ps[1 - player]),
            &self.inner.visible_pairs(),
            self.inner.legal_actions(player),
        )
    }

    fn visible_pairs(&self) -> Vec<(usize, usize)> {
        self.inner
            .visible_pairs()
            .into_iter()
            .map(pair_tuple)
            .collect()
    }

    /// プレイヤー `player` の自滅しない手のビットマスク(int)。予告おじゃまの降下は考えない。
    fn safe_actions(&self, player: usize) -> PyResult<u32> {
        check_player(player)?;
        Ok(self.inner.safe_actions(player))
    }

    /// `safe_actions` を長さ22の int8 配列で返す。
    fn safe_action_mask<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, PyArray1<i8>>> {
        check_player(player)?;
        Ok(PyArray1::from_vec(
            py,
            mask_vec(self.inner.safe_actions(player)),
        ))
    }

    /// プレイヤー `player` の手ごとの結果の特徴量 (22, NUM_ACTION_FEATURES) float32。
    /// 予告おじゃまの降下は考えない。
    fn action_features<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        check_player(player)?;
        let flat: Vec<f32> = self
            .inner
            .action_features(player)
            .iter()
            .flatten()
            .copied()
            .collect();
        PyArray1::from_vec(py, flat).reshape([NUM_ACTIONS, NUM_ACTION_FEATURES])
    }

    /// プレイヤー `player` の対戦用の手ごとの特徴量 (22, NUM_VERSUS_ACTION_FEATURES) float32。
    /// 列の意味は VERSUS_ACTION_FEATURE_NAMES(先頭13列は action_features と同じ)。
    fn versus_action_features<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        check_player(player)?;
        let flat: Vec<f32> = self
            .inner
            .versus_action_features(player)
            .iter()
            .flatten()
            .copied()
            .collect();
        PyArray1::from_vec(py, flat).reshape([NUM_ACTIONS, NUM_VERSUS_ACTION_FEATURES])
    }

    /// プレイヤー `player` から見た対局全体の情報 (NUM_VERSUS_INFO,) float32。
    /// 並びは VERSUS_INFO_NAMES。
    fn versus_info<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        check_player(player)?;
        Ok(PyArray1::from_vec(
            py,
            self.inner.versus_info(player).to_vec(),
        ))
    }

    /// プレイヤー `player` の潜在連鎖 (連鎖数, 得点)。
    fn potential_chain(&self, player: usize) -> PyResult<(u32, u32)> {
        check_player(player)?;
        let p = self.inner.potential_chain(player);
        Ok((p.chains, p.score))
    }

    /// 予告おじゃま数 [P0, P1]。
    fn pending_ojama(&self) -> (u32, u32) {
        (self.inner.player(0).pending, self.inner.player(1).pending)
    }

    fn scores(&self) -> (u64, u64) {
        (self.inner.player(0).score, self.inner.player(1).score)
    }

    #[getter]
    fn turn(&self) -> u64 {
        self.inner.turn()
    }

    #[getter]
    fn is_over(&self) -> bool {
        self.inner.is_over()
    }

    /// 勝者の番号。未決着・引き分けなら None。
    #[getter]
    fn winner(&self) -> Option<usize> {
        match self.inner.outcome() {
            Some(Outcome::Win(i)) => Some(i),
            _ => None,
        }
    }

    #[getter]
    fn is_draw(&self) -> bool {
        self.inner.outcome() == Some(Outcome::Draw)
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    fn field_str(&self, player: usize) -> PyResult<String> {
        check_player(player)?;
        Ok(self.inner.player(player).field.to_string())
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    fn __copy__(&self) -> Self {
        Clone::clone(self)
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        Clone::clone(self)
    }

    fn __repr__(&self) -> String {
        format!(
            "Versus(seed={}, turn={})",
            self.inner.seed(),
            self.inner.turn()
        )
    }
}

/// dict から時間の定数を読む(書かれていないキーは既定値)。
fn timing_from_dict(d: Option<&Bound<'_, PyDict>>) -> PyResult<Timing> {
    let mut t = Timing::default();
    let Some(d) = d else {
        return Ok(t);
    };
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "spawn" => t.spawn = v.extract()?,
            "drop_per_row" => t.drop_per_row = v.extract()?,
            "landing" => t.landing = v.extract()?,
            "chigiri_fall" => t.chigiri_fall = v.extract()?,
            "vanish" => t.vanish = v.extract()?,
            "chain_drop_per_row" => t.chain_drop_per_row = v.extract()?,
            "chain_landing" => t.chain_landing = v.extract()?,
            "ojama_landing" => t.ojama_landing = v.extract()?,
            "max_frames" => t.max_frames = v.extract()?,
            _ => return Err(PyValueError::new_err(format!("unknown timing key: {key}"))),
        }
    }
    Ok(t)
}

fn phase_name(p: &Phase) -> &'static str {
    match p {
        Phase::Ready => "ready",
        Phase::Placing { .. } => "placing",
        Phase::Chaining { .. } => "chaining",
        Phase::OjamaFalling { .. } => "ojama_falling",
        Phase::Dead => "dead",
    }
}

/// 2人対戦(時間のモデル)。エージェントが使うメソッドは `Versus` と同じ名前・引数。
///
/// 進め方: `needs_action()` が True のプレイヤーに `act(player, action)` で手を決めさせ、
/// `advance()` で次に誰かが操作待ちになるまで時刻を進める。
#[pyclass(name = "TimedVersus", module = "puyotrain._puyo", skip_from_py_object)]
#[derive(Clone)]
struct PyTimedVersus {
    inner: TimedVersus,
}

impl PyTimedVersus {
    /// 予告おじゃまを合計(確定 + 未確定)にしたプレイヤー状態(観測・対局全体の情報用)。
    fn total_state(&self, i: usize) -> PlayerState {
        let p = self.inner.player(i);
        let mut st = p.state;
        st.pending = p.total_pending();
        st
    }

    fn pairs2(&self, i: usize) -> [Pair; 2] {
        [self.inner.pair(i, 0), self.inner.pair(i, 1)]
    }
}

#[pymethods]
impl PyTimedVersus {
    /// `timing` は時間の定数の上書き(dict。キーは Timing のフィールド名)。
    #[new]
    #[pyo3(signature = (seed, timing=None))]
    fn new(seed: u64, timing: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(PyTimedVersus {
            inner: TimedVersus::with_timing(seed, timing_from_dict(timing)?),
        })
    }

    /// 手を決める必要があるプレイヤー (P0, P1)。
    fn needs_action(&self) -> (bool, bool) {
        let n = self.inner.needs_action();
        (n[0], n[1])
    }

    /// 操作待ちのプレイヤーが手を決める(時刻は進めない)。
    fn act(&mut self, player: usize, action_index: usize) -> PyResult<()> {
        check_player(player)?;
        self.inner
            .act(player, action(action_index)?)
            .map_err(to_py_err)
    }

    /// 誰かが操作待ちになるか、対局が終わるまで時刻を進める。
    fn advance(&mut self) {
        self.inner.advance()
    }

    /// `advance` と同じだが、時刻 `limit`(フレーム)を超える出来事は処理しない(描画用)。
    fn advance_until(&mut self, limit: u64) {
        self.inner.advance_until(limit)
    }

    /// 描画用の今の見た目: board (13, 6) int8、cleared (13, 6) bool か None(消去の演出中だけ)、
    /// chain(連鎖中なら何連鎖目か)、falling [(段(小数), 列, 色), ...](設置中の組ぷよ)、phase。
    fn display<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyDict>> {
        check_player(player)?;
        let disp = self.inner.display(player);
        let d = PyDict::new(py);
        d.set_item("board", grid(py, &disp.field)?)?;
        if disp.cleared != 0 {
            d.set_item("cleared", mask_grid(py, disp.cleared)?)?;
        } else {
            d.set_item("cleared", py.None())?;
        }
        d.set_item("chain", disp.chain)?;
        let falling: Vec<(f32, usize, usize)> = disp
            .falling
            .iter()
            .map(|&(row, col, color)| (row, col, color.index()))
            .collect();
        d.set_item("falling", falling)?;
        d.set_item("phase", phase_name(&self.inner.player(player).phase))?;
        Ok(d)
    }

    fn legal_actions(&self, player: usize) -> PyResult<u32> {
        check_player(player)?;
        Ok(self.inner.legal_actions(player))
    }

    fn action_mask<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, PyArray1<i8>>> {
        check_player(player)?;
        Ok(PyArray1::from_vec(
            py,
            mask_vec(self.inner.legal_actions(player)),
        ))
    }

    fn greedy_action(&self, player: usize, seed: u64) -> PyResult<Option<usize>> {
        check_player(player)?;
        Ok(self
            .inner
            .greedy_action(player, &mut Rng::new(seed))
            .map(Action::index))
    }

    fn safe_actions(&self, player: usize) -> PyResult<u32> {
        check_player(player)?;
        if !self.inner.needs_action()[player] {
            return Ok(0);
        }
        Ok(safe_actions(
            &self.inner.player(player).state.field,
            self.pairs2(player),
        ))
    }

    fn safe_action_mask<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, PyArray1<i8>>> {
        Ok(PyArray1::from_vec(py, mask_vec(self.safe_actions(player)?)))
    }

    /// プレイヤー `player` 視点の観測(`Versus.observation` と同じ形)。
    /// `pieces` は自分のツモの位置から、`ojama` は予告の合計(確定 + 未確定)。
    fn observation<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyDict>> {
        check_player(player)?;
        let me = self.total_state(player);
        let opp = self.total_state(1 - player);
        observation(
            py,
            &me,
            Some(&opp),
            &self.inner.visible_pairs(player),
            self.inner.legal_actions(player),
        )
    }

    fn action_features<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        check_player(player)?;
        let feats = if self.inner.needs_action()[player] {
            action_features(&self.inner.player(player).state.field, self.pairs2(player))
        } else {
            [[0.0; NUM_ACTION_FEATURES]; NUM_ACTIONS]
        };
        let flat: Vec<f32> = feats.iter().flatten().copied().collect();
        PyArray1::from_vec(py, flat).reshape([NUM_ACTIONS, NUM_ACTION_FEATURES])
    }

    /// このプレイヤーの観測を、重みの種類 `kind`(solo / versus / timed)に合わせて作る。
    /// `safe` なら行動マスクを自滅しない手に絞る(`TimedBatch.observe` の1局分と同じ)。
    #[pyo3(signature = (player, kind, safe=true))]
    fn observe<'py>(
        &self,
        py: Python<'py>,
        player: usize,
        kind: &str,
        safe: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        check_player(player)?;
        let kind = timed_batch::Kind::parse(kind)?;
        let o = timed_batch::observe_one(&self.inner, player, kind, safe);
        timed_batch::stack_observations(py, vec![o], kind, false)
    }

    /// 同期ターン制の対戦用の特徴量(`Versus.versus_action_features` と同じ20個)。
    /// 予告おじゃまは「次に着地したとき降る」確定分だけを使う。
    fn versus_action_features<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        check_player(player)?;
        let feats = if self.inner.needs_action()[player] {
            versus_action_features(&self.inner.player(player).state, self.pairs2(player))
        } else {
            [[0.0; NUM_VERSUS_ACTION_FEATURES]; NUM_ACTIONS]
        };
        let flat: Vec<f32> = feats.iter().flatten().copied().collect();
        PyArray1::from_vec(py, flat).reshape([NUM_ACTIONS, NUM_VERSUS_ACTION_FEATURES])
    }

    /// 対局全体の情報(`Versus.versus_info` と同じ8個。予告は確定 + 未確定の合計)。
    fn versus_info<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        check_player(player)?;
        let info = versus_info(&self.total_state(player), &self.total_state(1 - player));
        Ok(PyArray1::from_vec(py, info.to_vec()))
    }

    fn potential_chain(&self, player: usize) -> PyResult<(u32, u32)> {
        check_player(player)?;
        let p = potential_chain(&self.inner.player(player).state.field);
        Ok((p.chains, p.score))
    }

    fn visible_pairs(&self, player: usize) -> PyResult<Vec<(usize, usize)>> {
        check_player(player)?;
        Ok(self
            .inner
            .visible_pairs(player)
            .into_iter()
            .map(pair_tuple)
            .collect())
    }

    fn board<'py>(
        &self,
        py: Python<'py>,
        player: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray2<i8>>> {
        check_player(player)?;
        grid(py, &self.inner.player(player).state.field)
    }

    fn field_str(&self, player: usize) -> PyResult<String> {
        check_player(player)?;
        Ok(self.inner.player(player).state.field.to_string())
    }

    /// 予告おじゃま [(確定, 未確定) P0, P1]。
    fn pending_ojama(&self) -> ((u32, u32), (u32, u32)) {
        let p = |i: usize| {
            let pl = self.inner.player(i);
            (pl.state.pending, pl.unconfirmed)
        };
        (p(0), p(1))
    }

    /// 各プレイヤーの状態の名前 (ready / placing / chaining / ojama_falling / dead)。
    fn phases(&self) -> (&'static str, &'static str) {
        (
            phase_name(&self.inner.player(0).phase),
            phase_name(&self.inner.player(1).phase),
        )
    }

    fn scores(&self) -> (u64, u64) {
        (
            self.inner.player(0).state.score,
            self.inner.player(1).state.score,
        )
    }

    fn max_chains(&self) -> (u32, u32) {
        (
            self.inner.player(0).max_chain,
            self.inner.player(1).max_chain,
        )
    }

    fn moves(&self) -> (u64, u64) {
        (self.inner.player(0).moves, self.inner.player(1).moves)
    }

    /// 現在の時刻(フレーム)。
    #[getter]
    fn now(&self) -> u64 {
        self.inner.now()
    }

    /// 置いた手数の多い方(`Versus.turn` の代わり)。
    #[getter]
    fn turn(&self) -> u64 {
        self.inner.player(0).moves.max(self.inner.player(1).moves)
    }

    #[getter]
    fn is_over(&self) -> bool {
        self.inner.is_over()
    }

    #[getter]
    fn timed_out(&self) -> bool {
        self.inner.timed_out()
    }

    #[getter]
    fn winner(&self) -> Option<usize> {
        match self.inner.outcome() {
            Some(Outcome::Win(i)) => Some(i),
            _ => None,
        }
    }

    #[getter]
    fn is_draw(&self) -> bool {
        self.inner.outcome() == Some(Outcome::Draw)
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    /// 時間の定数(dict)。
    fn timing<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let t = self.inner.timing();
        let d = PyDict::new(py);
        d.set_item("spawn", t.spawn)?;
        d.set_item("drop_per_row", t.drop_per_row)?;
        d.set_item("landing", t.landing)?;
        d.set_item("chigiri_fall", t.chigiri_fall.to_vec())?;
        d.set_item("vanish", t.vanish)?;
        d.set_item("chain_drop_per_row", t.chain_drop_per_row)?;
        d.set_item("chain_landing", t.chain_landing)?;
        d.set_item("ojama_landing", t.ojama_landing.to_vec())?;
        d.set_item("max_frames", t.max_frames)?;
        Ok(d)
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    fn __repr__(&self) -> String {
        format!(
            "TimedVersus(seed={}, now={}, moves={:?})",
            self.inner.seed(),
            self.inner.now(),
            self.moves()
        )
    }
}

fn check_player(player: usize) -> PyResult<()> {
    if player < 2 {
        Ok(())
    } else {
        Err(PyValueError::new_err("player must be 0 or 1"))
    }
}

#[pymodule]
fn _puyo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyVersus>()?;
    m.add_class::<PyTimedVersus>()?;
    m.add_class::<timed_batch::PyTimedBatch>()?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("WIDTH", WIDTH)?;
    m.add("HEIGHT", HEIGHT)?;
    m.add("NUM_COLORS", NUM_COLORS)?;
    m.add("NUM_ACTION_FEATURES", NUM_ACTION_FEATURES)?;
    m.add("ACTION_FEATURE_NAMES", ACTION_FEATURE_NAMES.to_vec())?;
    m.add("NUM_VERSUS_ACTION_FEATURES", NUM_VERSUS_ACTION_FEATURES)?;
    m.add(
        "VERSUS_ACTION_FEATURE_NAMES",
        VERSUS_ACTION_FEATURE_NAMES.to_vec(),
    )?;
    m.add("NUM_VERSUS_INFO", NUM_VERSUS_INFO)?;
    m.add("VERSUS_INFO_NAMES", VERSUS_INFO_NAMES.to_vec())?;
    m.add("NUM_TIMED_ACTION_FEATURES", NUM_TIMED_ACTION_FEATURES)?;
    m.add(
        "TIMED_ACTION_FEATURE_NAMES",
        TIMED_ACTION_FEATURE_NAMES.to_vec(),
    )?;
    m.add("NUM_TIMED_INFO", NUM_TIMED_INFO)?;
    m.add("TIMED_INFO_NAMES", TIMED_INFO_NAMES.to_vec())?;
    Ok(())
}
