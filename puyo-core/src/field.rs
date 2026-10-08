//! 盤面。ぷよの種類ごとのビットボードで保持する。
//!
//! ビット配置: `bit = col * 16 + row`(col: 0..6 左から、row: 0..13 下から)。
//! 1列16bitのうち下位13bitを使う。6列で96bitなので `u128` 1本に収まる。
//! `Field` は `Copy` で、clone はメモリコピーのみ。

use std::fmt;

use crate::color::{Puyo, NUM_PLANES};
use crate::rules::{
    self, CLEAR_THRESHOLD, DEATH_COL, DEATH_ROW, HEIGHT, NUM_COLORS, VISIBLE_HEIGHT, WIDTH,
};

const COL_BITS: u32 = 16;
const COL_FULL: u128 = (1 << HEIGHT) - 1;
const COL_VISIBLE: u128 = (1 << VISIBLE_HEIGHT) - 1;

const fn repeat_cols(col_mask: u128) -> u128 {
    let mut m = 0;
    let mut c = 0;
    while c < WIDTH {
        m |= col_mask << (c as u32 * COL_BITS);
        c += 1;
    }
    m
}

/// 盤面全体(13段)のマスク。
pub const FULL_MASK: u128 = repeat_cols(COL_FULL);
/// 連結判定に使う領域(12段)のマスク。
pub const VISIBLE_MASK: u128 = repeat_cols(COL_VISIBLE);

#[inline]
const fn bit(col: usize, row: usize) -> u128 {
    1u128 << (col as u32 * COL_BITS + row as u32)
}

/// 上下左右への1マス膨張(盤外・13段目のマスクは呼び出し側で行う)。
#[inline]
fn expand(b: u128) -> u128 {
    b | (b << 1) | (b >> 1) | (b << COL_BITS) | (b >> COL_BITS)
}

/// 1回の消去判定の結果。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClearResult {
    /// 消えた色ぷよの数(おじゃまは含まない)。
    pub num_colored: u32,
    /// 消えたおじゃまの数。
    pub num_ojama: u32,
    /// 連結ボーナスの合計(グループごとに加算)。
    pub link_bonus: u32,
    /// 消えた色の数。
    pub num_colors: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Field {
    planes: [u128; NUM_PLANES],
}

impl Field {
    pub const fn new() -> Field {
        Field {
            planes: [0; NUM_PLANES],
        }
    }

    /// 文字列から盤面を作る(テスト・デバッグ用)。
    ///
    /// 1行が1段で、最終行が1段目。行は6文字で `R G B Y O` と `.`(空)。
    /// 行数は13以下。下詰めされていない盤面も作れる。
    pub fn from_rows(rows: &[&str]) -> Result<Field, String> {
        if rows.len() > HEIGHT {
            return Err(format!("too many rows: {}", rows.len()));
        }
        let mut f = Field::new();
        for (i, line) in rows.iter().rev().enumerate() {
            let chars: Vec<char> = line.chars().collect();
            if chars.len() != WIDTH {
                return Err(format!(
                    "row {} must have {} cells: {:?}",
                    i + 1,
                    WIDTH,
                    line
                ));
            }
            for (col, &c) in chars.iter().enumerate() {
                if c == '.' || c == ' ' {
                    continue;
                }
                let p = Puyo::from_char(c).ok_or_else(|| format!("unknown cell {:?}", c))?;
                f.set(col, i, Some(p));
            }
        }
        Ok(f)
    }

    /// 指定した種類のビットボード。
    #[inline]
    pub fn plane(&self, p: Puyo) -> u128 {
        self.planes[p.index()]
    }

    #[inline]
    pub fn occupied(&self) -> u128 {
        let mut o = 0;
        for p in &self.planes {
            o |= *p;
        }
        o
    }

    #[inline]
    pub fn get(&self, col: usize, row: usize) -> Option<Puyo> {
        debug_assert!(col < WIDTH && row < HEIGHT);
        let b = bit(col, row);
        self.planes
            .iter()
            .position(|p| p & b != 0)
            .and_then(Puyo::from_index)
    }

    #[inline]
    pub fn set(&mut self, col: usize, row: usize, puyo: Option<Puyo>) {
        assert!(col < WIDTH && row < HEIGHT, "cell out of range");
        let b = bit(col, row);
        for p in &mut self.planes {
            *p &= !b;
        }
        if let Some(p) = puyo {
            self.planes[p.index()] |= b;
        }
    }

    #[inline]
    fn column_bits(&self, col: usize) -> u32 {
        ((self.occupied() >> (col as u32 * COL_BITS)) & COL_FULL) as u32
    }

    /// 列の高さ(最上段のぷよの段数)。
    #[inline]
    pub fn height(&self, col: usize) -> usize {
        (32 - self.column_bits(col).leading_zeros()) as usize
    }

    #[inline]
    pub fn heights(&self) -> [usize; WIDTH] {
        let occ = self.occupied();
        let mut h = [0; WIDTH];
        for (c, hc) in h.iter_mut().enumerate() {
            let bits = ((occ >> (c as u32 * COL_BITS)) & COL_FULL) as u32;
            *hc = (32 - bits.leading_zeros()) as usize;
        }
        h
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.occupied() == 0
    }

    /// 窒息判定(左から3列目の12段目が埋まっているか)。
    #[inline]
    pub fn is_dead(&self) -> bool {
        self.occupied() & bit(DEATH_COL, DEATH_ROW) != 0
    }

    pub fn count(&self) -> u32 {
        self.occupied().count_ones()
    }

    /// 列の一番上にぷよを積む。13段を超える場合は置かずに `false` を返す。
    #[inline]
    pub fn push(&mut self, col: usize, puyo: Puyo) -> bool {
        let h = self.height(col);
        if h >= HEIGHT {
            return false;
        }
        self.planes[puyo.index()] |= bit(col, h);
        true
    }

    /// 浮いているぷよを落とす。何か動いたら `true`。
    pub fn apply_gravity(&mut self) -> bool {
        let occ = self.occupied();
        let mut moved = false;
        for c in 0..WIDTH {
            let shift = c as u32 * COL_BITS;
            let col_occ = ((occ >> shift) & COL_FULL) as u32;
            // 下から隙間なく詰まっていれば何もしない。
            if col_occ & (col_occ + 1) == 0 {
                continue;
            }
            moved = true;
            let col_mask = COL_FULL << shift;
            let mut cols = [0u32; NUM_PLANES];
            for (dst, p) in cols.iter_mut().zip(self.planes.iter()) {
                *dst = ((p >> shift) & COL_FULL) as u32;
            }
            let mut packed = [0u32; NUM_PLANES];
            let mut bits = col_occ;
            let mut k = 0;
            while bits != 0 {
                let r = bits.trailing_zeros();
                bits &= bits - 1;
                for (dst, src) in packed.iter_mut().zip(cols.iter()) {
                    *dst |= ((src >> r) & 1) << k;
                }
                k += 1;
            }
            for (p, src) in self.planes.iter_mut().zip(packed.iter()) {
                *p = (*p & !col_mask) | ((*src as u128) << shift);
            }
        }
        moved
    }

    /// 浮いたぷよがあるか(テスト用)。
    pub fn has_floating(&self) -> bool {
        (0..WIDTH).any(|c| {
            let b = self.column_bits(c);
            b & (b + 1) != 0
        })
    }

    /// 4個以上つながった色ぷよと、それに隣接するおじゃまを消す。
    /// 落下は行わない。何も消えなければ `None`。
    pub fn clear_groups(&mut self) -> Option<ClearResult> {
        let mut cleared: u128 = 0;
        let mut res = ClearResult::default();
        for c in 0..NUM_COLORS {
            let p = self.planes[c] & VISIBLE_MASK;
            if p.count_ones() < CLEAR_THRESHOLD {
                continue;
            }
            let mut color_used = false;
            let mut rem = p;
            while rem != 0 {
                let mut g = rem & rem.wrapping_neg();
                loop {
                    let n = expand(g) & p;
                    if n == g {
                        break;
                    }
                    g = n;
                }
                rem &= !g;
                let size = g.count_ones();
                if size >= CLEAR_THRESHOLD {
                    cleared |= g;
                    res.num_colored += size;
                    res.link_bonus += rules::link_bonus(size);
                    color_used = true;
                }
            }
            if color_used {
                res.num_colors += 1;
            }
        }
        if cleared == 0 {
            return None;
        }
        let ojama_idx = Puyo::Ojama.index();
        let ojama = expand(cleared) & self.planes[ojama_idx] & VISIBLE_MASK;
        res.num_ojama = ojama.count_ones();
        for p in &mut self.planes[..NUM_COLORS] {
            *p &= !cleared;
        }
        self.planes[ojama_idx] &= !ojama;
        Some(res)
    }

    /// 列ごとの個数だけおじゃまを積む。13段を超える分は消滅する。
    pub fn drop_ojama_columns(&mut self, counts: &[u32; WIDTH]) {
        let heights = self.heights();
        let idx = Puyo::Ojama.index();
        for c in 0..WIDTH {
            let h = heights[c];
            let n = (counts[c] as usize).min(HEIGHT - h);
            if n == 0 {
                continue;
            }
            let bits = ((1u128 << n) - 1) << h;
            self.planes[idx] |= bits << (c as u32 * COL_BITS);
        }
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in (0..HEIGHT).rev() {
            for col in 0..WIDTH {
                let c = self.get(col, row).map_or('.', Puyo::to_char);
                write!(f, "{}", c)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Field:")?;
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heights_and_push() {
        let mut f = Field::new();
        assert_eq!(f.height(0), 0);
        assert!(f.push(0, Puyo::Red));
        assert!(f.push(0, Puyo::Blue));
        assert_eq!(f.height(0), 2);
        assert_eq!(f.get(0, 1), Some(Puyo::Blue));
        for _ in 0..11 {
            assert!(f.push(0, Puyo::Green));
        }
        assert_eq!(f.height(0), 13);
        assert!(!f.push(0, Puyo::Green));
    }

    #[test]
    fn gravity_compacts() {
        let mut f = Field::from_rows(&["R.....", "......", "B.....", "......"]).unwrap();
        assert!(f.has_floating());
        assert!(f.apply_gravity());
        assert!(!f.has_floating());
        assert_eq!(f.get(0, 0), Some(Puyo::Blue));
        assert_eq!(f.get(0, 1), Some(Puyo::Red));
        assert_eq!(f.get(0, 2), None);
    }

    #[test]
    fn row13_excluded_exactly() {
        // 列0の 10..13段目が R(4個)だが 13段目は除外 → 3個で消えない。
        let mut f = Field::new();
        let fill = [Puyo::Green, Puyo::Blue, Puyo::Yellow];
        for i in 0..9 {
            f.push(0, fill[i % 3]);
        }
        for _ in 0..4 {
            f.push(0, Puyo::Red);
        }
        assert_eq!(f.height(0), 13);
        assert!(f.clear_groups().is_none());
        // 隣の列の12段目に R を足せば 4個(13段目除く)になり消える。
        for i in 0..11 {
            f.push(1, fill[(i + 1) % 3]);
        }
        f.push(1, Puyo::Red);
        let r = f.clear_groups().unwrap();
        assert_eq!(r.num_colored, 4);
        // 13段目の R は残る
        assert_eq!(f.get(0, 12), Some(Puyo::Red));
    }

    #[test]
    fn ojama_adjacent_cleared() {
        let mut f = Field::from_rows(&["O.....", "RO....", "RRRO..", "OOOOOO"]).unwrap();
        let r = f.clear_groups().unwrap();
        assert_eq!(r.num_colored, 4);
        // 隣接: (0,3) (1,2) (3,1) と、下の (0,0) (1,0) (2,0)
        assert_eq!(r.num_ojama, 6);
        f.apply_gravity();
        assert_eq!(f.count(), 3);
    }

    #[test]
    fn display_roundtrip() {
        let f = Field::from_rows(&["RG....", "BYO..."]).unwrap();
        let s = f.to_string();
        let rows: Vec<&str> = s.lines().collect();
        assert_eq!(Field::from_rows(&rows).unwrap(), f);
    }
}
