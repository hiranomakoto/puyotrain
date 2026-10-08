//! ルール定数。ぷよぷよ通準拠のパラメータはすべてこのファイルに集約する。

/// 盤面の列数。
pub const WIDTH: usize = 6;
/// 盤面の段数(13段目を含む)。
pub const HEIGHT: usize = 13;
/// 連結判定に含める段数(13段目は含めない)。
pub const VISIBLE_HEIGHT: usize = 12;

/// 通常色の数。
pub const NUM_COLORS: usize = 4;

/// 消去に必要な連結数。
pub const CLEAR_THRESHOLD: u32 = 4;

/// 窒息判定の列(0始まり。左から3列目)。
pub const DEATH_COL: usize = 2;
/// 窒息判定の段(0始まり。12段目)。
pub const DEATH_ROW: usize = 11;

/// ツモの出現列(0始まり。軸ぷよの列)。
pub const SPAWN_COL: usize = 2;
/// この高さ以上の列は越えられない(移動先にもできない)。
pub const BLOCKING_HEIGHT: usize = 12;

/// 1個消すごとの基本点。
pub const SCORE_PER_PUYO: u32 = 10;

/// 連鎖ボーナス(1連鎖目から)。表を超えた分は `CHAIN_BONUS_STEP` ずつ増える。
pub const CHAIN_BONUS: [u32; 7] = [0, 8, 16, 32, 64, 96, 128];
pub const CHAIN_BONUS_STEP: u32 = 32;

/// 連結ボーナス(連結数4から)。表を超えた分は `LINK_BONUS_MAX`。
pub const LINK_BONUS: [u32; 8] = [0, 2, 3, 4, 5, 6, 7, 10];
pub const LINK_BONUS_MAX: u32 = 10;

/// 色数ボーナス(1色から)。
pub const COLOR_BONUS: [u32; 4] = [0, 3, 6, 12];

/// おじゃまぷよ1個あたりの得点。
pub const SCORE_PER_OJAMA: u32 = 70;
/// 1回の降下で降るおじゃまの上限。
pub const MAX_OJAMA_DROP: u32 = 30;
/// 全消しボーナス(次の連鎖のおじゃまに上乗せする個数)。
pub const ALL_CLEAR_BONUS_OJAMA: u32 = 30;

/// 行動数(縦置き 6列×2向き + 横置き 5列×2向き)。
pub const NUM_ACTIONS: usize = 22;

/// 連鎖ボーナス(chain は1始まり)。
#[inline]
pub fn chain_bonus(chain: u32) -> u32 {
    let i = chain.saturating_sub(1) as usize;
    if i < CHAIN_BONUS.len() {
        CHAIN_BONUS[i]
    } else {
        CHAIN_BONUS[CHAIN_BONUS.len() - 1] + CHAIN_BONUS_STEP * (i - (CHAIN_BONUS.len() - 1)) as u32
    }
}

/// 連結ボーナス(size は4以上)。
#[inline]
pub fn link_bonus(size: u32) -> u32 {
    let i = size.saturating_sub(CLEAR_THRESHOLD) as usize;
    LINK_BONUS.get(i).copied().unwrap_or(LINK_BONUS_MAX)
}

/// 色数ボーナス(colors は1以上)。
#[inline]
pub fn color_bonus(colors: u32) -> u32 {
    let i = colors.saturating_sub(1) as usize;
    COLOR_BONUS
        .get(i)
        .copied()
        .unwrap_or(COLOR_BONUS[COLOR_BONUS.len() - 1])
}
