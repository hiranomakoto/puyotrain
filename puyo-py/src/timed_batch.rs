//! 時間のモデルの観測(1局分)と、複数局をまとめて進める `TimedBatch`。
//!
//! 観測の種類(`kind`)は、使う重みが学習したときの入力に合わせる:
//!
//! - `"solo"`: 1人用の手ごとの特徴量(13個)
//! - `"versus"`: 同期ターン制の対戦用(手ごと20個 + `versus_info` 8個。降る予告は確定分)
//! - `"timed"`: 時間のモデル用(手ごと22個 + `versus_info` 17個)
//!
//! `safe` なら行動マスクを、手ごとの特徴量の legal・dead・survivable から作った
//! 「自滅しない手」に絞る(該当が無ければ合法手のまま。Python の `_with_options` と同じ)。

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use puyo_core::afterstate::{
    action_features, versus_action_features, versus_info, NUM_ACTION_FEATURES,
    NUM_VERSUS_ACTION_FEATURES, NUM_VERSUS_INFO,
};
use puyo_core::rules::NUM_ACTIONS;
use puyo_core::timed::{TimedVersus, Timing};
use puyo_core::timed_features::{
    timed_action_features, timed_info, NUM_TIMED_ACTION_FEATURES, NUM_TIMED_INFO,
};
use puyo_core::{Outcome, PlayerState, Rng};

use super::{
    action, check_player, mask_vec, pieces_vec, timing_from_dict, to_py_err, write_board,
    BOARD_LEN, NUM_BOARD_PLANES, NUM_VISIBLE_PAIRS,
};
use puyo_core::rules::{HEIGHT, NUM_COLORS, WIDTH};

/// 観測の種類。
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Solo,
    Versus,
    Timed,
}

impl Kind {
    pub(crate) fn parse(s: &str) -> PyResult<Kind> {
        match s {
            "solo" => Ok(Kind::Solo),
            "versus" => Ok(Kind::Versus),
            "timed" => Ok(Kind::Timed),
            _ => Err(PyValueError::new_err(format!(
                "kind must be solo / versus / timed: {s}"
            ))),
        }
    }

    fn n_features(self) -> usize {
        match self {
            Kind::Solo => NUM_ACTION_FEATURES,
            Kind::Versus => NUM_VERSUS_ACTION_FEATURES,
            Kind::Timed => NUM_TIMED_ACTION_FEATURES,
        }
    }

    fn n_info(self) -> usize {
        match self {
            Kind::Solo => 0,
            Kind::Versus => NUM_VERSUS_INFO,
            Kind::Timed => NUM_TIMED_INFO,
        }
    }
}

/// 1局・1プレイヤー分の観測(平らな配列)。
pub(crate) struct ObsOne {
    board: Vec<u8>,
    pieces: Vec<u8>,
    ojama: [i32; 2],
    mask: Vec<i8>,
    features: Vec<f32>,
    info: Vec<f32>,
}

/// 予告を合計(確定 + 未確定)にしたプレイヤー状態。
fn total_state(g: &TimedVersus, i: usize) -> PlayerState {
    let p = g.player(i);
    let mut st = p.state;
    st.pending = p.total_pending();
    st
}

pub(crate) fn observe_one(g: &TimedVersus, i: usize, kind: Kind, safe: bool) -> ObsOne {
    let me = total_state(g, i);
    let opp = total_state(g, 1 - i);
    let mut board = vec![0u8; 2 * BOARD_LEN];
    write_board(&mut board[..BOARD_LEN], &me);
    write_board(&mut board[BOARD_LEN..], &opp);
    let ready = g.needs_action()[i];
    let pairs = [g.pair(i, 0), g.pair(i, 1)];
    let features: Vec<f32> = if !ready {
        vec![0.0; NUM_ACTIONS * kind.n_features()]
    } else {
        match kind {
            Kind::Solo => action_features(&g.player(i).state.field, pairs)
                .iter()
                .flatten()
                .copied()
                .collect(),
            Kind::Versus => versus_action_features(&g.player(i).state, pairs)
                .iter()
                .flatten()
                .copied()
                .collect(),
            Kind::Timed => timed_action_features(g, i)
                .iter()
                .flatten()
                .copied()
                .collect(),
        }
    };
    let info = match kind {
        Kind::Solo => Vec::new(),
        Kind::Versus => versus_info(&me, &opp).to_vec(),
        Kind::Timed => timed_info(g, i).to_vec(),
    };
    let mut mask = mask_vec(g.legal_actions(i));
    if safe && ready {
        // legal(0)・dead(1)・survivable(10)。並びはどの種類でも同じ
        let n = kind.n_features();
        let safe_mask: Vec<i8> = (0..NUM_ACTIONS)
            .map(|a| {
                let r = &features[a * n..(a + 1) * n];
                (r[0] > 0.0 && r[1] == 0.0 && r[10] > 0.0) as i8
            })
            .collect();
        if safe_mask.iter().any(|&x| x != 0) {
            mask = safe_mask;
        }
    }
    ObsOne {
        board,
        pieces: pieces_vec(&g.visible_pairs(i)),
        ojama: [me.pending as i32, opp.pending as i32],
        mask,
        features,
        info,
    }
}

/// 観測(複数)を numpy 配列の dict にする。`batched` なら先頭に件数の次元を付け、
/// そうでなければ1件分の形(`Versus.observation` と同じ)にする。
pub(crate) fn stack_observations<'py>(
    py: Python<'py>,
    obs: Vec<ObsOne>,
    kind: Kind,
    batched: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let n = obs.len();
    let shape = |rest: &[usize]| -> Vec<usize> {
        if batched {
            std::iter::once(n).chain(rest.iter().copied()).collect()
        } else {
            rest.to_vec()
        }
    };
    let mut board = Vec::with_capacity(n * 2 * BOARD_LEN);
    let mut pieces = Vec::with_capacity(n * NUM_VISIBLE_PAIRS * 2 * NUM_COLORS);
    let mut ojama = Vec::with_capacity(n * 2);
    let mut mask = Vec::with_capacity(n * NUM_ACTIONS);
    let mut features = Vec::with_capacity(n * NUM_ACTIONS * kind.n_features());
    let mut info = Vec::with_capacity(n * kind.n_info());
    for o in obs {
        board.extend_from_slice(&o.board);
        pieces.extend_from_slice(&o.pieces);
        ojama.extend_from_slice(&o.ojama);
        mask.extend_from_slice(&o.mask);
        features.extend_from_slice(&o.features);
        info.extend_from_slice(&o.info);
    }
    let d = PyDict::new(py);
    d.set_item(
        "board",
        PyArray1::from_vec(py, board).reshape(shape(&[2, NUM_BOARD_PLANES, HEIGHT, WIDTH]))?,
    )?;
    d.set_item(
        "pieces",
        PyArray1::from_vec(py, pieces).reshape(shape(&[NUM_VISIBLE_PAIRS, 2, NUM_COLORS]))?,
    )?;
    d.set_item("ojama", PyArray1::from_vec(py, ojama).reshape(shape(&[2]))?)?;
    d.set_item(
        "action_mask",
        PyArray1::from_vec(py, mask).reshape(shape(&[NUM_ACTIONS]))?,
    )?;
    d.set_item(
        "action_features",
        PyArray1::from_vec(py, features).reshape(shape(&[NUM_ACTIONS, kind.n_features()]))?,
    )?;
    if kind != Kind::Solo {
        d.set_item(
            "versus_info",
            PyArray1::from_vec(py, info).reshape(shape(&[kind.n_info()]))?,
        )?;
    }
    Ok(d)
}

/// 時間のモデルの対局を複数まとめて持ち、観測の計算と時刻の進行を並列に行う。
///
/// 対局の番号 `idx` とプレイヤー `players` を並べて渡す。観測は件数の次元を先頭に持つ
/// numpy 配列の dict(1局分の `TimedVersus.observe` を積み重ねたものと同じ)。
#[pyclass(name = "TimedBatch", module = "puyotrain._puyo", skip_from_py_object)]
pub(crate) struct PyTimedBatch {
    games: Vec<TimedVersus>,
    timing: Timing,
}

impl PyTimedBatch {
    fn check(&self, idx: &[usize]) -> PyResult<()> {
        if idx.iter().any(|&i| i >= self.games.len()) {
            return Err(PyValueError::new_err("game index out of range"));
        }
        Ok(())
    }
}

#[pymethods]
impl PyTimedBatch {
    /// `n` 局分(シード0)で作る。`reset(i, seed)` で各局を始め直す。
    #[new]
    #[pyo3(signature = (n, timing=None))]
    fn new(n: usize, timing: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let timing = timing_from_dict(timing)?;
        Ok(PyTimedBatch {
            games: vec![TimedVersus::with_timing(0, timing); n],
            timing,
        })
    }

    fn __len__(&self) -> usize {
        self.games.len()
    }

    /// 対局 `i` をシード `seed` で始め直す。
    fn reset(&mut self, i: usize, seed: u64) -> PyResult<()> {
        self.check(&[i])?;
        self.games[i] = TimedVersus::with_timing(seed, self.timing);
        Ok(())
    }

    /// 手を決める必要があるか (対局数, 2) bool。
    fn needs_action<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<bool>>> {
        let v: Vec<bool> = self.games.iter().flat_map(|g| g.needs_action()).collect();
        PyArray1::from_vec(py, v).reshape([self.games.len(), 2])
    }

    fn is_over<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.games.iter().map(TimedVersus::is_over).collect())
    }

    /// 勝者 (対局数,) int: 0 / 1、引き分けは 2、未決着・打ち切りは −1。
    fn winners<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let v = self
            .games
            .iter()
            .map(|g| match g.outcome() {
                Some(Outcome::Win(i)) => i as i64,
                Some(Outcome::Draw) => 2,
                None => -1,
            })
            .collect();
        PyArray1::from_vec(py, v)
    }

    fn timed_out<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.games.iter().map(TimedVersus::timed_out).collect())
    }

    /// 現在の時刻(フレーム) (対局数,)。
    fn now<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_vec(py, self.games.iter().map(TimedVersus::now).collect())
    }

    /// 各プレイヤーの統計 (対局数, 2): 置いた手数・最大連鎖・生んだおじゃまの合計・得点。
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let n = self.games.len();
        let col = |f: &dyn Fn(&puyo_core::timed::TimedPlayer) -> u64| -> Vec<u64> {
            self.games
                .iter()
                .flat_map(|g| [f(g.player(0)), f(g.player(1))])
                .collect()
        };
        let d = PyDict::new(py);
        d.set_item(
            "moves",
            PyArray1::from_vec(py, col(&|p| p.moves)).reshape([n, 2])?,
        )?;
        d.set_item(
            "max_chain",
            PyArray1::from_vec(py, col(&|p| p.max_chain as u64)).reshape([n, 2])?,
        )?;
        d.set_item(
            "attack_total",
            PyArray1::from_vec(py, col(&|p| p.attack_total)).reshape([n, 2])?,
        )?;
        d.set_item(
            "score",
            PyArray1::from_vec(py, col(&|p| p.state.score)).reshape([n, 2])?,
        )?;
        Ok(d)
    }

    /// 対局 `idx[k]` のプレイヤー `players[k]` の潜在連鎖数。
    fn potential_chains(&self, idx: Vec<usize>, players: Vec<usize>) -> PyResult<Vec<u32>> {
        self.check(&idx)?;
        Ok(idx
            .iter()
            .zip(&players)
            .map(|(&i, &p)| {
                puyo_core::potential::potential_chain(&self.games[i].player(p).state.field).chains
            })
            .collect())
    }

    /// 対局 `idx[k]` のプレイヤー `players[k]` が手 `actions[k]` を決める。
    fn act(&mut self, idx: Vec<usize>, players: Vec<usize>, actions: Vec<usize>) -> PyResult<()> {
        self.check(&idx)?;
        for ((&i, &p), &a) in idx.iter().zip(&players).zip(&actions) {
            check_player(p)?;
            self.games[i].act(p, action(a)?).map_err(to_py_err)?;
        }
        Ok(())
    }

    /// 対局 `idx` の時刻を、誰かが操作待ちになるか終わるまで進める(並列)。
    fn advance(&mut self, py: Python<'_>, idx: Vec<usize>) -> PyResult<()> {
        self.check(&idx)?;
        let mut selected = vec![false; self.games.len()];
        for &i in &idx {
            selected[i] = true;
        }
        let games = &mut self.games;
        py.detach(|| {
            games
                .par_iter_mut()
                .zip(selected.par_iter())
                .filter(|(_, &s)| s)
                .for_each(|(g, _)| g.advance());
        });
        Ok(())
    }

    /// 観測をまとめて計算する(並列)。`kind` は solo / versus / timed。
    #[pyo3(signature = (idx, players, kind, safe=true))]
    fn observe<'py>(
        &self,
        py: Python<'py>,
        idx: Vec<usize>,
        players: Vec<usize>,
        kind: &str,
        safe: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.check(&idx)?;
        let kind = Kind::parse(kind)?;
        let games = &self.games;
        let obs: Vec<ObsOne> = py.detach(|| {
            idx.par_iter()
                .zip(players.par_iter())
                .map(|(&i, &p)| observe_one(&games[i], p, kind, safe))
                .collect()
        });
        stack_observations(py, obs, kind, true)
    }

    /// greedy(2手読み)の手をまとめて求める(並列)。`seeds[k]` は同点の手を選ぶ乱数のシード。
    fn greedy(
        &self,
        py: Python<'_>,
        idx: Vec<usize>,
        players: Vec<usize>,
        seeds: Vec<u64>,
    ) -> PyResult<Vec<usize>> {
        self.check(&idx)?;
        let games = &self.games;
        let acts: Vec<Option<usize>> = py.detach(|| {
            idx.par_iter()
                .zip(players.par_iter())
                .zip(seeds.par_iter())
                .map(|((&i, &p), &s)| {
                    games[i]
                        .greedy_action(p, &mut Rng::new(s))
                        .map(puyo_core::Action::index)
                })
                .collect()
        });
        acts.into_iter()
            .map(|a| a.ok_or_else(|| PyValueError::new_err("player is not waiting for an action")))
            .collect()
    }

    /// 対局 `i` の複製(`TimedVersus`)。
    fn game(&self, i: usize) -> PyResult<super::PyTimedVersus> {
        self.check(&[i])?;
        Ok(super::PyTimedVersus {
            inner: self.games[i],
        })
    }
}
