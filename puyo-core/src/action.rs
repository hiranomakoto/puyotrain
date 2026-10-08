//! 行動(配置位置と向き)と合法手マスク。
//!
//! 番号付け(22通り):
//! - 0..6   : 縦置き、子ぷよが上。軸の列 = i
//! - 6..12  : 縦置き、子ぷよが下。軸の列 = i - 6
//! - 12..17 : 横置き、子ぷよが右。軸の列 = i - 12(子は +1 列)
//! - 17..22 : 横置き、子ぷよが左。軸の列 = i - 16(子は -1 列)

use crate::color::Pair;
use crate::field::Field;
use crate::rules::{BLOCKING_HEIGHT, NUM_ACTIONS, SPAWN_COL, WIDTH};

/// 子ぷよの向き。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rotation {
    Up,
    Right,
    Down,
    Left,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Action(u8);

/// 合法手のビットマスク(bit i が行動 i)。
pub type ActionMask = u32;

impl Action {
    pub const COUNT: usize = NUM_ACTIONS;

    #[inline]
    pub fn new(index: usize) -> Option<Action> {
        (index < NUM_ACTIONS).then_some(Action(index as u8))
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn all() -> impl Iterator<Item = Action> {
        (0..NUM_ACTIONS as u8).map(Action)
    }

    /// 軸の列と向きから作る。盤外なら `None`。
    pub fn from_placement(axis_col: usize, rot: Rotation) -> Option<Action> {
        let i = match rot {
            Rotation::Up if axis_col < WIDTH => axis_col,
            Rotation::Down if axis_col < WIDTH => 6 + axis_col,
            Rotation::Right if axis_col + 1 < WIDTH => 12 + axis_col,
            Rotation::Left if (1..WIDTH).contains(&axis_col) => 16 + axis_col,
            _ => return None,
        };
        Some(Action(i as u8))
    }

    #[inline]
    pub fn axis_col(self) -> usize {
        let i = self.0 as usize;
        match i {
            0..=5 => i,
            6..=11 => i - 6,
            12..=16 => i - 12,
            _ => i - 16,
        }
    }

    #[inline]
    pub fn rotation(self) -> Rotation {
        match self.0 {
            0..=5 => Rotation::Up,
            6..=11 => Rotation::Down,
            12..=16 => Rotation::Right,
            _ => Rotation::Left,
        }
    }

    #[inline]
    pub fn child_col(self) -> usize {
        match self.rotation() {
            Rotation::Up | Rotation::Down => self.axis_col(),
            Rotation::Right => self.axis_col() + 1,
            Rotation::Left => self.axis_col() - 1,
        }
    }

    /// 移動で通過する列の範囲(出現列と配置列を含む)。
    #[inline]
    fn span(self) -> (usize, usize) {
        let a = self.axis_col();
        let c = self.child_col();
        (a.min(c).min(SPAWN_COL), a.max(c).max(SPAWN_COL))
    }
}

/// 列の高さから「その列に入れるか」のビットを作る(bit c: 列 c の高さ < 12)。
#[inline]
fn open_columns(field: &Field) -> u32 {
    let h = field.heights();
    let mut open = 0;
    for (c, &hc) in h.iter().enumerate() {
        if hc < BLOCKING_HEIGHT {
            open |= 1 << c;
        }
    }
    open
}

/// 合法手マスク。出現列から配置列までの間(両端含む)に高さ12以上の列があれば不可。
pub fn legal_actions(field: &Field) -> ActionMask {
    let open = open_columns(field);
    let mut mask = 0;
    for a in Action::all() {
        let (lo, hi) = a.span();
        let need = ((1u32 << (hi + 1)) - 1) & !((1u32 << lo) - 1);
        if open & need == need {
            mask |= 1 << a.index();
        }
    }
    mask
}

/// 組ぷよを置く(落下込み)。合法性は呼び出し側で確認すること。
/// 合法手なら13段を超えることはない。
#[inline]
pub fn place_pair(field: &mut Field, pair: Pair, action: Action) {
    match action.rotation() {
        Rotation::Up => {
            field.push(action.axis_col(), pair.axis);
            field.push(action.axis_col(), pair.child);
        }
        Rotation::Down => {
            field.push(action.axis_col(), pair.child);
            field.push(action.axis_col(), pair.axis);
        }
        Rotation::Right | Rotation::Left => {
            field.push(action.axis_col(), pair.axis);
            field.push(action.child_col(), pair.child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Puyo;

    #[test]
    fn roundtrip() {
        for a in Action::all() {
            assert_eq!(Action::from_placement(a.axis_col(), a.rotation()), Some(a));
            assert!(a.child_col() < WIDTH);
        }
        assert_eq!(Action::all().count(), 22);
    }

    #[test]
    fn empty_field_all_legal() {
        assert_eq!(legal_actions(&Field::new()), (1 << 22) - 1);
    }

    #[test]
    fn blocked_column_blocks_beyond() {
        let mut f = Field::new();
        for _ in 0..12 {
            f.push(1, Puyo::Ojama); // 2列目を高さ12に
        }
        let m = legal_actions(&f);
        for a in Action::all() {
            let legal = m & (1 << a.index()) != 0;
            let touches_left = a.axis_col() <= 1 || a.child_col() <= 1;
            assert_eq!(legal, !touches_left, "{:?}", a);
        }
    }

    #[test]
    fn place_vertical_and_horizontal() {
        let mut f = Field::new();
        let p = Pair::new(Puyo::Red, Puyo::Blue);
        place_pair(
            &mut f,
            p,
            Action::from_placement(0, Rotation::Down).unwrap(),
        );
        assert_eq!(f.get(0, 0), Some(Puyo::Blue));
        assert_eq!(f.get(0, 1), Some(Puyo::Red));
        place_pair(
            &mut f,
            p,
            Action::from_placement(1, Rotation::Left).unwrap(),
        );
        assert_eq!(f.get(1, 0), Some(Puyo::Red));
        assert_eq!(f.get(0, 2), Some(Puyo::Blue));
    }
}
