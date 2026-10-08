//! ぷよの種類と組ぷよ。

use crate::rules::NUM_COLORS;

/// ぷよの種類。値は `Field` 内部のプレーン番号と一致する。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Puyo {
    Red = 0,
    Green = 1,
    Blue = 2,
    Yellow = 3,
    Ojama = 4,
}

/// プレーン数(通常色 + おじゃま)。
pub const NUM_PLANES: usize = NUM_COLORS + 1;

impl Puyo {
    /// 通常色のみ。
    pub const COLORS: [Puyo; NUM_COLORS] = [Puyo::Red, Puyo::Green, Puyo::Blue, Puyo::Yellow];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn from_index(i: usize) -> Option<Puyo> {
        match i {
            0 => Some(Puyo::Red),
            1 => Some(Puyo::Green),
            2 => Some(Puyo::Blue),
            3 => Some(Puyo::Yellow),
            4 => Some(Puyo::Ojama),
            _ => None,
        }
    }

    #[inline]
    pub fn is_color(self) -> bool {
        self != Puyo::Ojama
    }

    /// 盤面表記用の1文字(R/G/B/Y/O)。
    pub fn to_char(self) -> char {
        match self {
            Puyo::Red => 'R',
            Puyo::Green => 'G',
            Puyo::Blue => 'B',
            Puyo::Yellow => 'Y',
            Puyo::Ojama => 'O',
        }
    }

    pub fn from_char(c: char) -> Option<Puyo> {
        match c.to_ascii_uppercase() {
            'R' => Some(Puyo::Red),
            'G' => Some(Puyo::Green),
            'B' => Some(Puyo::Blue),
            'Y' => Some(Puyo::Yellow),
            'O' => Some(Puyo::Ojama),
            _ => None,
        }
    }
}

/// 組ぷよ。`axis` が軸ぷよ、`child` が回転する子ぷよ。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pair {
    pub axis: Puyo,
    pub child: Puyo,
}

impl Pair {
    pub const fn new(axis: Puyo, child: Puyo) -> Pair {
        Pair { axis, child }
    }
}
