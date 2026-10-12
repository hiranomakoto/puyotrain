//! 時間のモデル用の特徴量(手ごと・対局全体)。
//!
//! 手ごとの特徴量の先頭20個は、同期ターン制の対戦用([`crate::afterstate::versus_action_features`])と
//! 同じ並び・同じ意味で、予告おじゃまの扱いだけ時間のモデルに合わせる
//! (相殺は確定 + 未確定の合計に対して、降るのは確定分だけ)。続く2個が時間の情報。
//! 対局全体の情報の先頭8個は [`crate::afterstate::versus_info`] と同じ。

use crate::action::{legal_actions, place_pair, Action};
use crate::afterstate::{
    action_features, drop_ojama_evenly, log_score, ojama_scale, versus_info, NUM_ACTION_FEATURES,
    NUM_VERSUS_ACTION_FEATURES, NUM_VERSUS_INFO,
};
use crate::chain::step_score;
use crate::field::Field;
use crate::potential::potential_chain;
use crate::rules::{MAX_OJAMA_DROP, NUM_ACTIONS, SCORE_PER_OJAMA};
use crate::score::attack_ojama;
use crate::timed::{Phase, TimedVersus, Timing};

/// 時間のモデル用の1手あたりの特徴量の数(対戦用の20個 + 時間の2個)。
pub const NUM_TIMED_ACTION_FEATURES: usize = NUM_VERSUS_ACTION_FEATURES + 2;

/// 手ごとの特徴量の名前(並び順。先頭20個は対戦用と同じ)。
pub const TIMED_ACTION_FEATURE_NAMES: [&str; NUM_TIMED_ACTION_FEATURES] = [
    "legal",
    "dead",
    "chains",
    "log_score",
    "potential_chains",
    "potential_log_score",
    "max_height",
    "death_col_height",
    "count",
    "adjacency",
    "survivable",
    "next_best_chains",
    "next_best_log_score",
    "ojama_offset",
    "ojama_sent",
    "ojama_dropped",
    "pending_after",
    "dead_after_drop",
    "potential_chains_after_drop",
    "potential_log_score_after_drop",
    "action_seconds",
    "finishes_before_opp_chain",
];

/// 時間のモデル用の対局全体の情報の数(対戦用の8個 + 9個)。
pub const NUM_TIMED_INFO: usize = NUM_VERSUS_INFO + 9;

/// 対局全体の情報の名前(並び順。先頭8個は対戦用と同じ。予告は確定 + 未確定の合計)。
pub const TIMED_INFO_NAMES: [&str; NUM_TIMED_INFO] = [
    "my_pending",
    "opp_pending",
    "my_potential_chains",
    "my_potential_log_score",
    "opp_potential_chains",
    "opp_potential_log_score",
    "opp_max_height",
    "opp_death_col_height",
    "my_confirmed",
    "my_unconfirmed",
    "opp_chaining",
    "opp_links_remaining",
    "opp_chain_ojama",
    "opp_chain_end_seconds",
    "opp_busy_seconds",
    "opp_placing",
    "opp_ojama_falling",
];

/// 秒を 0〜数 程度に収める(10秒で割る)。
#[inline]
fn seconds(frames: u64) -> f32 {
    frames as f32 / 60.0 / 10.0
}

/// 連鎖の続き(`chain` 連鎖目まで解決済みの盤面から)を最後まで解いた見込み。
/// 戻り値: (残りの連鎖数, 残りの得点, 残りの連鎖にかかるフレーム数)。
fn remaining_chain(field: &Field, chain: u32, timing: &Timing) -> (u32, u32, u64) {
    let mut f = *field;
    let (mut links, mut score, mut frames) = (0u32, 0u32, 0u64);
    while let Some(clear) = f.clear_groups() {
        links += 1;
        score += step_score(chain + links, &clear);
        frames += timing.link_frames(&f) as u64;
        f.apply_gravity();
    }
    (links, score, frames)
}

/// 相手 `j` の連鎖の見込み: (連鎖中か, 残りの連鎖数, 送ってくるおじゃまの見込み,
/// 連鎖が終わるまでのフレーム数)。連鎖中でなければ (false, 0, 0, 0)。
/// 送ってくる量は、相手の端数を含めて換算し、相手自身の予告との相殺は考えない。
fn opponent_chain(g: &TimedVersus, j: usize) -> (bool, u32, u32, u64) {
    let o = g.player(j);
    match o.phase {
        Phase::Chaining { until, chain } => {
            let (links, score, frames) = remaining_chain(&o.state.field, chain, g.timing());
            let ojama = (score + o.state.carry) / SCORE_PER_OJAMA;
            (true, links, ojama, until - g.now() + frames)
        }
        _ => (false, 0, 0, 0),
    }
}

/// プレイヤー `i` から見た対局全体の情報。並びは [`TIMED_INFO_NAMES`]。
pub fn timed_info(g: &TimedVersus, i: usize) -> [f32; NUM_TIMED_INFO] {
    let j = 1 - i;
    let (me, opp) = (g.player(i), g.player(j));
    let mut me_total = me.state;
    me_total.pending = me.total_pending();
    let mut opp_total = opp.state;
    opp_total.pending = opp.total_pending();
    let base = versus_info(&me_total, &opp_total);

    let (chaining, links, ojama, chain_end) = opponent_chain(g, j);
    let busy = match opp.phase {
        Phase::Chaining { .. } => chain_end,
        Phase::Placing { until, .. } | Phase::OjamaFalling { until } => until - g.now(),
        Phase::Ready | Phase::Dead | Phase::Idle => 0,
    };
    let mut out = [0.0; NUM_TIMED_INFO];
    out[..NUM_VERSUS_INFO].copy_from_slice(&base);
    out[8] = ojama_scale(me.state.pending);
    out[9] = ojama_scale(me.unconfirmed);
    out[10] = chaining as u8 as f32;
    out[11] = links as f32 / 10.0;
    out[12] = ojama_scale(ojama);
    out[13] = seconds(chain_end);
    out[14] = seconds(busy);
    out[15] = matches!(opp.phase, Phase::Placing { .. }) as u8 as f32;
    out[16] = matches!(opp.phase, Phase::OjamaFalling { .. }) as u8 as f32;
    out
}

/// プレイヤー `i`(操作待ち)の手ごとの特徴量。並びは [`TIMED_ACTION_FEATURE_NAMES`]。
///
/// - 先頭13個は [`action_features`] と同じ。
/// - ojama_offset / ojama_sent: この手の連鎖が生むおじゃまで、予告(確定 + 未確定)を相殺する量と、
///   残りを相手に送る量(/30)。連鎖1つずつ、端数の持ち越しと全消しボーナスを含めて計算する。
/// - ojama_dropped: 連鎖しなかった場合に、着地後に降る確定分(/30、最大30個)。
/// - pending_after: この手の後に残る予告(確定 + 未確定、/30)。
/// - dead_after_drop, potential_*_after_drop: おじゃまが降った後の窒息と潜在連鎖
///   (降る位置は [`drop_ojama_evenly`] で近似)。
/// - action_seconds: この手にかかる時間(設置 + 起きる連鎖。10秒で割る)。
/// - finishes_before_opp_chain: 相手が連鎖中で、この手(設置 + 連鎖)が相手の連鎖より先に終わるなら1。
///
/// 操作待ちでない、または合法でない手は全部0。
pub fn timed_action_features(
    g: &TimedVersus,
    i: usize,
) -> [[f32; NUM_TIMED_ACTION_FEATURES]; NUM_ACTIONS] {
    let mut out = [[0.0; NUM_TIMED_ACTION_FEATURES]; NUM_ACTIONS];
    if !g.needs_action()[i] {
        return out;
    }
    let p = g.player(i);
    let st = &p.state;
    let pairs = [g.pair(i, 0), g.pair(i, 1)];
    let base = action_features(&st.field, pairs);
    let timing = g.timing();
    let (opp_chaining, _, _, opp_chain_end) = opponent_chain(g, 1 - i);
    let total = p.total_pending();

    let mut m = legal_actions(&st.field);
    while m != 0 {
        let a = Action::new(m.trailing_zeros() as usize).unwrap();
        m &= m - 1;
        let row = &mut out[a.index()];
        row[..NUM_ACTION_FEATURES].copy_from_slice(&base[a.index()]);

        let mut frames = timing.place_frames(&st.field, a) as u64;
        let mut field = st.field;
        place_pair(&mut field, pairs[0], a);
        let mut carry = st.carry;
        let (mut chain, mut attack) = (0u32, 0u32);
        while let Some(clear) = field.clear_groups() {
            chain += 1;
            frames += timing.link_frames(&field) as u64;
            field.apply_gravity();
            let bonus = st.all_clear && chain == 1;
            attack += attack_ojama(step_score(chain, &clear), &mut carry, bonus);
        }
        let (offset, sent, dropped, after) = if chain > 0 {
            let offset = attack.min(total);
            (offset, attack - offset, 0, field)
        } else {
            let dropped = st.pending.min(MAX_OJAMA_DROP);
            (0, 0, dropped, drop_ojama_evenly(&field, dropped))
        };
        let dead = after.is_dead();
        row[13] = ojama_scale(offset);
        row[14] = ojama_scale(sent);
        row[15] = ojama_scale(dropped);
        row[16] = ojama_scale(total - offset - dropped);
        row[17] = dead as u8 as f32;
        if !dead {
            let pot = potential_chain(&after);
            row[18] = pot.chains as f32 / 10.0;
            row[19] = log_score(pot.score);
        }
        row[20] = seconds(frames);
        row[21] = (opp_chaining && frames < opp_chain_end) as u8 as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Rotation;
    use crate::color::Puyo;

    fn up(col: usize) -> Action {
        Action::from_placement(col, Rotation::Up).unwrap()
    }

    #[test]
    fn info_at_start_is_quiet() {
        let g = TimedVersus::new(1);
        let info = timed_info(&g, 0);
        assert!(info.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn action_features_extend_base_and_time_placement() {
        let g = TimedVersus::new(2);
        let f = timed_action_features(&g, 0);
        let base = action_features(&Field::new(), [g.pair(0, 0), g.pair(0, 1)]);
        for (a, row) in f.iter().enumerate() {
            assert_eq!(&row[..NUM_ACTION_FEATURES], &base[a]);
        }
        // 空の列への縦置き: 54F
        assert_eq!(f[up(0).index()][20], 54.0 / 60.0 / 10.0);
        assert_eq!(f[up(0).index()][21], 0.0);
    }

    #[test]
    fn opponent_chain_is_forecast_and_compared() {
        let mut g = TimedVersus::new(3);
        // P1 は連鎖中(1連鎖目を解決済み)。盤面に残りの2連鎖が仕込んである
        g.player_mut(1).state.field =
            Field::from_rows(&["G.....", "RR....", "RGGG..", "R....."]).unwrap();
        // ↑ R が4個つながっている(列0の下3 + 列1)ので、もう1つ目で消え、G が落ちてつながる
        g.player_mut(1).phase = Phase::Chaining {
            until: 30,
            chain: 1,
        };
        let info = timed_info(&g, 0);
        assert_eq!(info[10], 1.0);
        assert_eq!(info[11], 0.2); // 残り2連鎖
        assert!(info[13] > 30.0 / 60.0 / 10.0);
        assert_eq!(info[14], info[13]);
        // 手ごとの「相手の連鎖より先に終わるか」: 空の列に置くだけ(54F)は終わる時刻(30F + 2連鎖)より前
        let f = timed_action_features(&g, 0);
        assert_eq!(f[up(0).index()][21], 1.0);
    }

    #[test]
    fn offset_uses_total_and_drop_uses_confirmed() {
        let mut g = TimedVersus::new(4);
        let c = g.pair(0, 0).axis.to_char();
        let row = format!("{c}.....");
        g.player_mut(0).state.field = Field::from_rows(&[&row, &row, &row]).unwrap();
        g.player_mut(0).state.carry = 69; // 端数込みで1個以上になる
        g.player_mut(0).state.pending = 6;
        g.player_mut(0).unconfirmed = 12;
        let f = timed_action_features(&g, 0);
        let clear = f[up(0).index()];
        assert_eq!(clear[2], 0.1);
        // 生んだ分(子ぷよの色しだいで1〜2個)をすべて相殺に使い、送らない。降らない
        assert!(clear[13] >= 1.0 / 30.0);
        assert_eq!(clear[14], 0.0);
        assert_eq!(clear[15], 0.0);
        assert!((clear[16] - (18.0 / 30.0 - clear[13])).abs() < 1e-6);
        let quiet = f[up(5).index()];
        assert_eq!(quiet[15], 6.0 / 30.0); // 降るのは確定分だけ
        assert_eq!(quiet[16], 12.0 / 30.0); // 未確定は残る
        assert_eq!(Puyo::COLORS.len(), 4);
    }
}
