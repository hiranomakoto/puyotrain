//! 手ごとの結果の特徴量。22通りの手それぞれについて、置いた後の盤面を実際に計算して
//! 「その手を打つと何が起きるか」を数値にする。方策ネットが手ごとに点数をつけるのに使う。

use crate::action::{legal_actions, place_pair, Action};
use crate::chain::{resolve_chain, ChainSummary};
use crate::color::Pair;
use crate::field::Field;
use crate::phase::{place_and_resolve, PlayerState};
use crate::potential::potential_chain;
use crate::rules::{DEATH_COL, HEIGHT, MAX_OJAMA_DROP, NUM_ACTIONS, VISIBLE_HEIGHT, WIDTH};

/// 1手あたりの特徴量の数。
pub const NUM_ACTION_FEATURES: usize = 13;

/// 特徴量の名前(並び順)。
pub const ACTION_FEATURE_NAMES: [&str; NUM_ACTION_FEATURES] = [
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
];

/// 得点を 0〜1 程度に収める(ln(1 + 10万点) ≒ 11.5)。
#[inline]
pub(crate) fn log_score(score: u32) -> f32 {
    (score as f32).ln_1p() / 12.0
}

#[inline]
fn simulate(field: &Field, pair: Pair, action: Action) -> (Field, ChainSummary) {
    let mut f = *field;
    place_pair(&mut f, pair, action);
    let s = resolve_chain(&mut f);
    (f, s)
}

/// 手ごとの特徴量。合法でない手は全部0。
///
/// - legal: 合法手なら1
/// - dead: 置いた結果窒息するなら1
/// - chains, log_score: この手で起きる連鎖数(/10)と得点(対数)
/// - potential_chains, potential_log_score: 置いた後の潜在連鎖(/10、得点は対数)。窒息なら0
/// - max_height, death_col_height: 置いた後の最大の高さと窒息列の高さ(/13)
/// - count: 置いた後のぷよの数(/72)
/// - adjacency: 置いた後の同色の隣接数(/100)
/// - survivable: ネクストを窒息せずに置ける手があるなら1
/// - next_best_chains, next_best_log_score: 置いた後、ネクストで起こせる最大の連鎖(/10)と得点(対数)
pub fn action_features(
    field: &Field,
    pairs: [Pair; 2],
) -> [[f32; NUM_ACTION_FEATURES]; NUM_ACTIONS] {
    let mut out = [[0.0; NUM_ACTION_FEATURES]; NUM_ACTIONS];
    let mut m1 = legal_actions(field);
    while m1 != 0 {
        let a1 = Action::new(m1.trailing_zeros() as usize).unwrap();
        m1 &= m1 - 1;
        let (f1, s1) = simulate(field, pairs[0], a1);
        let dead = f1.is_dead();
        let heights = f1.heights();
        let row = &mut out[a1.index()];
        row[0] = 1.0;
        row[1] = dead as u8 as f32;
        row[2] = s1.chains as f32 / 10.0;
        row[3] = log_score(s1.score);
        row[6] = *heights.iter().max().unwrap() as f32 / HEIGHT as f32;
        row[7] = heights[DEATH_COL] as f32 / HEIGHT as f32;
        row[8] = f1.count() as f32 / (VISIBLE_HEIGHT * WIDTH) as f32;
        row[9] = f1.same_color_adjacency() as f32 / 100.0;
        if dead {
            continue;
        }
        let p = potential_chain(&f1);
        row[4] = p.chains as f32 / 10.0;
        row[5] = log_score(p.score);

        let mut survivable = false;
        let mut best = (0u32, 0u32);
        let mut m2 = legal_actions(&f1);
        while m2 != 0 {
            let a2 = Action::new(m2.trailing_zeros() as usize).unwrap();
            m2 &= m2 - 1;
            let (f2, s2) = simulate(&f1, pairs[1], a2);
            survivable |= !f2.is_dead();
            best = best.max((s2.chains, s2.score));
        }
        row[10] = survivable as u8 as f32;
        row[11] = best.0 as f32 / 10.0;
        row[12] = log_score(best.1);
    }
    out
}

/// 対戦用の1手あたりの特徴量の数(1人用の13個 + 予告おじゃまの7個)。
pub const NUM_VERSUS_ACTION_FEATURES: usize = NUM_ACTION_FEATURES + 7;

/// 対戦用の特徴量の名前(並び順。先頭13個は [`ACTION_FEATURE_NAMES`] と同じ)。
pub const VERSUS_ACTION_FEATURE_NAMES: [&str; NUM_VERSUS_ACTION_FEATURES] = [
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
];

/// 対局全体の情報の数。
pub const NUM_VERSUS_INFO: usize = 8;

/// 対局全体の情報の名前(並び順)。
pub const VERSUS_INFO_NAMES: [&str; NUM_VERSUS_INFO] = [
    "my_pending",
    "opp_pending",
    "my_potential_chains",
    "my_potential_log_score",
    "opp_potential_chains",
    "opp_potential_log_score",
    "opp_max_height",
    "opp_death_col_height",
];

/// おじゃまの個数を 0〜1 程度に収める(1回の降下の上限30個で割る)。
#[inline]
pub(crate) fn ojama_scale(n: u32) -> f32 {
    n as f32 / MAX_OJAMA_DROP as f32
}

/// おじゃま `n` 個を均等に降らせた盤面(特徴量用の近似)。
///
/// 実際の降下では、6で割った端数の列は乱数で決まる。ここでは窒息列(左から3列目)から
/// 中央寄りの順に1個ずつ置く(窒息の判定が甘くならないよう、窒息列を最初にする)。
pub fn drop_ojama_evenly(field: &Field, n: u32) -> Field {
    const ORDER: [usize; WIDTH] = [DEATH_COL, 3, 1, 4, 0, 5];
    let w = WIDTH as u32;
    let mut counts = [n / w; WIDTH];
    for &c in ORDER.iter().take((n % w) as usize) {
        counts[c] += 1;
    }
    let mut f = *field;
    f.drop_ojama_columns(&counts);
    f
}

/// 対戦用の手ごとの特徴量。先頭13個は [`action_features`] と同じ(予告おじゃまを考えない)。
/// 続く7個は、自分の予告おじゃま(`st.pending`)を考えたこのターンの結果:
///
/// - ojama_offset: この手の連鎖で相殺できる予告おじゃまの数(/30)
/// - ojama_sent: 相殺した後に相手へ送る数(/30)。相手が同じターンに撃つ分は考えない
/// - ojama_dropped: 連鎖しなかった場合にこのターン降ってくる数(/30)
/// - pending_after: このターンの後に残る予告おじゃまの数(/30)
/// - dead_after_drop: おじゃまが降った後に窒息しているなら1(降らない場合は dead と同じ)
/// - potential_chains_after_drop, potential_log_score_after_drop: 降った後の潜在連鎖
///
/// 降る位置は [`drop_ojama_evenly`] で近似する。合法でない手は全部0。
pub fn versus_action_features(
    st: &PlayerState,
    pairs: [Pair; 2],
) -> [[f32; NUM_VERSUS_ACTION_FEATURES]; NUM_ACTIONS] {
    let base = action_features(&st.field, pairs);
    let mut out = [[0.0; NUM_VERSUS_ACTION_FEATURES]; NUM_ACTIONS];
    let mut m = legal_actions(&st.field);
    while m != 0 {
        let a = Action::new(m.trailing_zeros() as usize).unwrap();
        m &= m - 1;
        let row = &mut out[a.index()];
        row[..NUM_ACTION_FEATURES].copy_from_slice(&base[a.index()]);

        let mut s = *st;
        let p = place_and_resolve(&mut s, pairs[0], a);
        let (offset, sent, dropped, after) = if p.chain.chains > 0 {
            let offset = p.attack.min(st.pending);
            (offset, p.attack - offset, 0, s.field)
        } else {
            let dropped = st.pending.min(MAX_OJAMA_DROP);
            (0, 0, dropped, drop_ojama_evenly(&s.field, dropped))
        };
        let dead = after.is_dead();
        row[13] = ojama_scale(offset);
        row[14] = ojama_scale(sent);
        row[15] = ojama_scale(dropped);
        row[16] = ojama_scale(st.pending - offset - dropped);
        row[17] = dead as u8 as f32;
        if !dead {
            let pot = potential_chain(&after);
            row[18] = pot.chains as f32 / 10.0;
            row[19] = log_score(pot.score);
        }
    }
    out
}

/// 対局全体の情報(自分 `me` と相手 `opp` の状態から)。並びは [`VERSUS_INFO_NAMES`]。
pub fn versus_info(me: &PlayerState, opp: &PlayerState) -> [f32; NUM_VERSUS_INFO] {
    let mine = potential_chain(&me.field);
    let theirs = potential_chain(&opp.field);
    let h = opp.field.heights();
    [
        ojama_scale(me.pending),
        ojama_scale(opp.pending),
        mine.chains as f32 / 10.0,
        log_score(mine.score),
        theirs.chains as f32 / 10.0,
        log_score(theirs.score),
        *h.iter().max().unwrap() as f32 / HEIGHT as f32,
        h[DEATH_COL] as f32 / HEIGHT as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Puyo;

    fn player(rows: &[&str], pending: u32) -> PlayerState {
        let mut st = PlayerState::new(0);
        st.field = Field::from_rows(rows).unwrap();
        st.pending = pending;
        st
    }

    #[test]
    fn versus_features_extend_solo_features() {
        let st = player(&["R.....", "R.....", "R....."], 0);
        let base = action_features(&st.field, [RG, BY]);
        let vs = versus_action_features(&st, [RG, BY]);
        for (b, v) in base.iter().zip(vs.iter()) {
            assert_eq!(&v[..NUM_ACTION_FEATURES], b);
            // 予告おじゃま0なら、降る・残る分は0で、降った後は置いた直後と同じ
            assert_eq!(v[15], 0.0);
            assert_eq!(v[16], 0.0);
            assert_eq!(v[17], v[1]);
            assert_eq!(v[18], v[4]);
        }
    }

    #[test]
    fn clearing_move_offsets_pending_and_quiet_move_takes_drop() {
        // 予告おじゃま12個。R を下にした縦置きで列0に置くと1連鎖(40点 → 0個)、
        // 消さない手では12個(2段)降ってくる
        let st = player(&["R.....", "R.....", "R....."], 12);
        let vs = versus_action_features(&st, [RG, BY]);
        let clear = vs.iter().find(|r| r[2] > 0.0).unwrap();
        assert_eq!(clear[15], 0.0); // 連鎖したので降らない
        assert_eq!(clear[16], 12.0 / 30.0); // 攻撃0個なので予告は残る
        let quiet = vs.iter().find(|r| r[0] == 1.0 && r[2] == 0.0).unwrap();
        assert_eq!(quiet[15], 12.0 / 30.0);
        assert_eq!(quiet[16], 0.0);
    }

    #[test]
    fn big_drop_kills() {
        // 窒息列が10段。30個(5段)降ると窒息する
        let rows = [
            "..R...", "..G...", "..B...", "..Y...", "..R...", "..G...", "..B...", "..Y...",
            "..R...", "..G...",
        ];
        let st = player(&rows, 30);
        let vs = versus_action_features(&st, [RG, BY]);
        let legal_quiet: Vec<_> = vs.iter().filter(|r| r[0] == 1.0 && r[2] == 0.0).collect();
        assert!(!legal_quiet.is_empty());
        assert!(legal_quiet.iter().all(|r| r[17] == 1.0));
    }

    #[test]
    fn even_drop_puts_remainder_on_death_column_first() {
        let f = drop_ojama_evenly(&Field::new(), 7);
        let h = f.heights();
        assert_eq!(h[DEATH_COL], 2);
        assert_eq!(h.iter().sum::<usize>(), 7);
    }

    #[test]
    fn versus_info_reads_both_players() {
        let me = player(&["R.....", "R.....", "R....."], 6);
        let opp = player(&["..Y...", "..Y..."], 0);
        let info = versus_info(&me, &opp);
        assert_eq!(info[0], 6.0 / 30.0);
        assert_eq!(info[1], 0.0);
        assert_eq!(info[2], 0.1);
        assert_eq!(info[4], 0.1);
        assert_eq!(info[7], 2.0 / 13.0);
    }

    const RG: Pair = Pair::new(Puyo::Red, Puyo::Green);
    const BY: Pair = Pair::new(Puyo::Blue, Puyo::Yellow);

    #[test]
    fn illegal_actions_are_zero_and_legal_flag_matches() {
        let f = Field::new();
        let feats = action_features(&f, [RG, BY]);
        let legal = legal_actions(&f);
        for (i, row) in feats.iter().enumerate() {
            assert_eq!(row[0] == 1.0, legal >> i & 1 == 1);
            if legal >> i & 1 == 0 {
                assert!(row.iter().all(|&x| x == 0.0));
            } else {
                assert_eq!(row[8], 2.0 / 72.0);
                assert_eq!(row[10], 1.0);
            }
        }
    }

    #[test]
    fn clearing_move_reports_chain() {
        // 列0 に R が3個。R を下にした縦置きで列0に置くと消える
        let f = Field::from_rows(&["R.....", "R.....", "R....."]).unwrap();
        let feats = action_features(&f, [RG, BY]);
        let best = feats.iter().map(|r| r[2]).fold(0.0, f32::max);
        assert_eq!(best, 0.1);
    }

    #[test]
    fn potential_after_placement() {
        // 列0 に R を2個置く手は潜在連鎖1(もう1個 R を足せば消える)を作る
        let f = Field::from_rows(&["R.....", "R....."]).unwrap();
        let feats = action_features(&f, [Pair::new(Puyo::Green, Puyo::Green), BY]);
        assert!(feats.iter().any(|r| r[4] == 0.1));
    }
}
