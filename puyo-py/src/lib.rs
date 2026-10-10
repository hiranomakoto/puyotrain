//! puyo-core の Python バインディング(モジュール名 `puyotrain._puyo`)。

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use puyo_core::afterstate::{ACTION_FEATURE_NAMES, NUM_ACTION_FEATURES};
use puyo_core::field::FULL_MASK;
use puyo_core::rules::{HEIGHT, NUM_ACTIONS, NUM_COLORS, WIDTH};
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
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("WIDTH", WIDTH)?;
    m.add("HEIGHT", HEIGHT)?;
    m.add("NUM_COLORS", NUM_COLORS)?;
    m.add("NUM_ACTION_FEATURES", NUM_ACTION_FEATURES)?;
    m.add("ACTION_FEATURE_NAMES", ACTION_FEATURE_NAMES.to_vec())?;
    Ok(())
}
