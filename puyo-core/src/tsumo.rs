//! シード指定可能な乱数とツモ列。
//!
//! ツモは `(seed, index)` から決定的に計算するため、列を保持しない。
//! 両プレイヤーは同じ seed を共有し、同じ順番で同じ組ぷよを受け取る。

use crate::color::{Pair, Puyo};
use crate::rules::NUM_COLORS;

/// SplitMix64 の出力関数。
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// SplitMix64 乱数生成器。状態は u64 1つで Copy。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub const fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    /// [0, n) の一様乱数(乗算シフト法。n は小さい前提で偏りは無視できる)。
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        (((self.next_u64() >> 32) * n as u64) >> 32) as u32
    }
}

/// ツモ列。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tsumo {
    seed: u64,
}

impl Tsumo {
    pub const fn new(seed: u64) -> Tsumo {
        Tsumo { seed }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// `index` 番目の組ぷよ。
    #[inline]
    pub fn pair(&self, index: u64) -> Pair {
        let r = mix64(self.seed ^ mix64(index.wrapping_add(0x5851_f42d_4c95_7f2d)));
        let n = NUM_COLORS as u64;
        let axis = Puyo::COLORS[(((r & 0xffff_ffff) * n) >> 32) as usize];
        let child = Puyo::COLORS[(((r >> 32) * n) >> 32) as usize];
        Pair::new(axis, child)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_balanced() {
        let t = Tsumo::new(42);
        assert_eq!(t.pair(10), Tsumo::new(42).pair(10));
        let mut counts = [0u32; NUM_COLORS];
        for i in 0..40_000 {
            let p = t.pair(i);
            counts[p.axis.index()] += 1;
            counts[p.child.index()] += 1;
        }
        for c in counts {
            assert!((19_000..21_000).contains(&c), "{:?}", counts);
        }
    }

    #[test]
    fn rng_below_range() {
        let mut r = Rng::new(1);
        for _ in 0..1000 {
            assert!(r.below(6) < 6);
        }
    }
}
