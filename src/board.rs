#![warn(clippy::nursery)]
// #![warn(clippy::restriction)]

use cli_table::{Style, Table};
use std::fmt;
use std::ops::{Index, IndexMut};
// use std::ops::Index;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Board(pub [[Cell; 9]; 9]);

impl Board {
    /// A `Board` is `solved` when all the rows, columns and houses contain the numbers from 1 to 9

    pub fn is_solved(&self) -> bool {
        for i in 0..9 {
            // check if all the numbers are present in the rows
            {
                let mut required_nums = ConstraintList::full();
                for index in RowIter::new(i) {
                    let cell = self[index];
                    match cell {
                        Cell::Number(n) => required_nums.remove(n),
                        Cell::Free => return false,
                    };
                }
                if !required_nums.is_empty() {
                    return false;
                }
            }

            // check if all the numbers are present in the columns
            {
                let mut required_nums = ConstraintList::full();
                for index in ColIter::new(i) {
                    let cell = self[index];
                    if let Cell::Number(n) = cell {
                        required_nums.remove(n);
                    }
                }
                if !required_nums.is_empty() {
                    return false;
                }
            }

            // check if all the numbers are present in the squares
            {
                let mut required_nums = ConstraintList::full();
                for index in BoxIter::new(i) {
                    let cell = self[index];
                    if let Cell::Number(n) = cell {
                        required_nums.remove(n);
                    }
                }
                if !required_nums.is_empty() {
                    return false;
                }
            }
        }

        true
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum Cell {
    Number(SudokuNum),
    Free,
}

// stores the row_index in the upper 4 bits and the col_index in the lower 4 bits
#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct BoardIndex(u8);

impl BoardIndex {
    #[inline]
    pub const fn new(row_index: u8, col_index: u8) -> Self {
        debug_assert!(row_index <= 8 && col_index <= 8);
        let index = (row_index << 4) | col_index;
        Self(index)
    }
}

impl Index<BoardIndex> for Board {
    type Output = Cell;

    fn index(&self, index: BoardIndex) -> &Self::Output {
        let row_index = ((index.0 & 0b_1111_0000) >> 4) as usize;
        let col_index = (index.0 & 0b_0000_1111) as usize;
        debug_assert!(row_index <= 8 && col_index <= 8);
        unsafe { self.0.get_unchecked(row_index).get_unchecked(col_index) }
    }
}

#[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
pub struct BigBoardIndex {
    pub row_index: usize,
    pub col_index: usize,
}

impl BigBoardIndex {
    #[inline(always)]
    pub const fn new(row_index: usize, col_index: usize) -> Self {
        Self { row_index, col_index }
    }
}

impl Index<BigBoardIndex> for Board {
    type Output = Cell;

    #[inline(always)]
    fn index(&self, index: BigBoardIndex) -> &Self::Output {
        debug_assert!(index.row_index <= 8 && index.col_index <= 8);
        unsafe { self.0.get_unchecked(index.row_index).get_unchecked(index.col_index) }
    }
}

impl IndexMut<BigBoardIndex> for Board {
    #[inline(always)]
    fn index_mut(&mut self, index: BigBoardIndex) -> &mut Self::Output {
        debug_assert!(index.row_index <= 8 && index.col_index <= 8);
        unsafe {
            self.0
                .get_unchecked_mut(index.row_index)
                .get_unchecked_mut(index.col_index)
        }
    }
}

pub struct BoardWithConstraints(pub [[CellWithConstraints; 9]; 9]);

impl BoardWithConstraints {
    #[inline(always)]
    pub const fn new() -> Self {
        Self([[CellWithConstraints::Free; 9]; 9])
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum CellWithConstraints {
    Number(SudokuNum),
    Constrained(ConstraintList),
    Free,
}

impl fmt::Display for CellWithConstraints {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::Constrained(cons) => write!(f, "{cons:?}"),
            Self::Free => write!(f, "."),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::EnumCount)]
pub enum SudokuNum {
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
}

impl std::cmp::PartialEq<u8> for SudokuNum {
    #[inline(always)]
    fn eq(&self, other: &u8) -> bool {
        (*self as u8) == *other
    }
}

impl std::fmt::Display for SudokuNum {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", *self as u8)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InvalidSudokuNumError;

impl TryFrom<usize> for SudokuNum {
    type Error = InvalidSudokuNumError;

    #[inline(always)]
    fn try_from(number: usize) -> Result<Self, InvalidSudokuNumError> {
        match number {
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            5 => Ok(Self::Five),
            6 => Ok(Self::Six),
            7 => Ok(Self::Seven),
            8 => Ok(Self::Eight),
            9 => Ok(Self::Nine),
            _ => Err(InvalidSudokuNumError),
        }
    }
}

impl From<u8> for SudokuNum {
    #[inline(always)]
    fn from(number: u8) -> Self {
        (number as usize)
            .try_into()
            .expect("failed to convert to sudoku number")
    }
}

impl From<SudokuNum> for ConstraintList {
    #[inline(always)]
    fn from(num: SudokuNum) -> Self {
        let mut res = Self::empty();
        res.insert(num);
        res
    }
}

#[derive(Hash, Copy, Clone, Eq, PartialEq)]
pub struct ConstraintList(pub U9BitArray);

impl ConstraintList {
    #[inline(always)]
    pub const fn full() -> Self {
        Self(U9BitArray::new(0b_0000_0001_1111_1111))
    }

    #[inline(always)]
    pub const fn empty() -> Self {
        Self(U9BitArray::new(0b_0000_0000_0000_0000))
    }

    #[inline(always)]
    const fn from_raw_bits(raw: u16) -> Self {
        Self(U9BitArray::new(raw))
    }

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub fn naked_single(self) -> Option<SudokuNum> {
        // PERF(Simon): maybe set hint to llvm that the first branch is far more likely
        if self.len() != 1 {
            return None;
        }
        let num = ((self.0.first_index() + 1) as u8)
            .try_into()
            .expect("failed to convert to sudoku number");
        Some(num)
    }

    #[inline(always)]
    pub fn insert(&mut self, num: SudokuNum) {
        self.0.set_bit((num as u8) - 1);
    }

    #[inline(always)]
    pub fn remove(&mut self, num: SudokuNum) {
        self.0.clear_bit((num as u8) - 1);
    }

    #[inline(always)]
    pub fn remove_all(&mut self, other: Self) {
        self.0.mask(other.0);
    }

    #[inline(always)]
    pub const fn contains(self, needle: SudokuNum) -> bool {
        self.0.is_bit_set((needle as usize) - 1)
    }

    #[inline(always)]
    pub const fn contains_all(self, other: Self) -> bool {
        self.0 .0 & other.0 .0 == other.0 .0
    }

    #[inline(always)]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub const fn combinations(self, k: u8) -> CombinationsIter {
        let current = (1 << k) - 1;
        CombinationsIter {
            k,
            bits: self.0 .0,
            current,
        }
    }

    #[inline(always)]
    pub const fn intersection(c0: Self, c1: Self, c2: Self) -> Self {
        Self::from_raw_bits(c0.0 .0 & c1.0 .0 & c2.0 .0)
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let table = self.0.table().bold(true).display().unwrap();

        write!(f, "\n{table}\n")
    }
}

impl fmt::Debug for BoardWithConstraints {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let table = self.0.table().bold(true).display().unwrap();

        write!(f, "\n{table}\n")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[rustc_layout_scalar_valid_range_end(0b0000_00001_1111_1111)]
#[rustc_layout_scalar_valid_range_start(0)]
pub struct U9BitArray(u16);

impl U9BitArray {
    #[inline(always)]
    pub const fn new(value: u16) -> Self {
        debug_assert!(value <= 0b0000_0001_1111_1111);
        unsafe { Self(value) }
    }

    #[inline(always)]
    pub fn set_bit(&mut self, index: u8) {
        assert!(index <= 9, "index out of range");
        unsafe {
            self.0 |= 1 << index;
        }
    }

    #[inline(always)]
    pub const fn is_bit_set(self, index: usize) -> bool {
        assert!(index <= 9, "index out of range");
        self.0 & (1 << index) != 0
    }

    #[inline(always)]
    pub fn clear_bit(&mut self, index: u8) {
        assert!(index <= 9, "index out of range");
        unsafe {
            self.0 &= !(1 << index);
        }
    }

    #[inline(always)]
    pub fn mask(&mut self, other: Self) {
        unsafe {
            self.0 &= !other.0;
        }
    }

    #[inline(always)]
    pub const fn count_ones(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub fn first_index(self) -> u32 {
        assert_ne!(self.0, 0, "not bits set");
        self.0.trailing_zeros()
    }
}

impl std::fmt::Debug for U9BitArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:016b}", self.0)
    }
}

const MIDDLE_OF_SQUARE_INDEXES: [(usize, usize); 9] =
    [(1, 1), (1, 4), (1, 7), (4, 1), (4, 4), (4, 7), (7, 1), (7, 4), (7, 7)];

const MIDDLE_OF_SQUARE_OFFSETS: [(isize, isize); 9] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

pub struct BoardIter {
    row_index: usize,
    col_index: usize,
}

impl BoardIter {
    pub const fn new() -> Self {
        Self {
            row_index: 0,
            col_index: 0,
        }
    }
}

impl Iterator for BoardIter {
    type Item = BigBoardIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row_index >= 9 {
            return None;
        }

        let result = BigBoardIndex::new(self.row_index, self.col_index);

        self.col_index += 1;
        if self.col_index >= 9 {
            self.col_index = 0;
            self.row_index += 1;
        }

        Some(result)
    }
}

pub struct RowIter {
    row: usize,
    cursor: usize,
}

impl RowIter {
    pub const fn new(row: usize) -> Self {
        debug_assert!(row <= 8);
        Self { row, cursor: 0 }
    }
}

impl Iterator for RowIter {
    type Item = BigBoardIndex;

    fn next(&mut self) -> Option<Self::Item> {
        let cursor = self.cursor;
        self.cursor += 1;

        if cursor > 8 {
            return None;
        }

        Some(BigBoardIndex::new(self.row, cursor))
    }
}

pub struct ColIter {
    col: usize,
    cursor: usize,
}

impl ColIter {
    pub const fn new(col: usize) -> Self {
        debug_assert!(col <= 8);
        Self { col, cursor: 0 }
    }
}

impl Iterator for ColIter {
    type Item = BigBoardIndex;

    fn next(&mut self) -> Option<Self::Item> {
        let cursor = self.cursor;
        self.cursor += 1;

        if cursor > 8 {
            return None;
        }

        Some(BigBoardIndex::new(cursor, self.col))
    }
}

pub struct BoxIter {
    row_index: usize,
    col_index: usize,
    cursor: usize,
}

impl BoxIter {
    pub const fn new(square: usize) -> Self {
        debug_assert!(square <= 8);
        let (row_index, col_index) = MIDDLE_OF_SQUARE_INDEXES[square];
        Self {
            row_index,
            col_index,
            cursor: 0,
        }
    }
}

impl Iterator for BoxIter {
    type Item = BigBoardIndex;

    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_wrap)]
    fn next(&mut self) -> Option<Self::Item> {
        let cursor = self.cursor;
        self.cursor += 1;

        if cursor >= MIDDLE_OF_SQUARE_INDEXES.len() {
            return None;
        }

        let (row_offset, col_offset) = MIDDLE_OF_SQUARE_OFFSETS[cursor];
        let row_index = ((self.row_index as isize) + row_offset) as usize;
        let col_index = ((self.col_index as isize) + col_offset) as usize;

        Some(BigBoardIndex::new(row_index, col_index))
    }
}

pub struct CombinationsIter {
    k: u8,
    bits: u16,
    current: u16,
}

impl Iterator for CombinationsIter {
    type Item = ConstraintList;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current <= self.bits {
            let k: u32 = self.k.into();
            if self.current & self.bits == self.current && self.current.count_ones() == k {
                let result = self.current;
                let tmp = self.current & (!self.current + 1);
                let mobile = self.current + tmp;
                self.current = (((mobile ^ self.current) >> 2) / tmp) | mobile;
                return Some(ConstraintList::from_raw_bits(result));
            }
            let tmp = self.current & (!self.current + 1);
            let mobile = self.current + tmp;
            self.current = (((mobile ^ self.current) >> 2) / tmp) | mobile;
        }
        None
    }
}

pub struct ConstraintListIter {
    bits: U9BitArray,
    cursor: u8,
}

impl Iterator for ConstraintListIter {
    type Item = SudokuNum;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let cursor = self.cursor;
            self.cursor += 1;

            if cursor > 8 {
                return None;
            }
            if self.bits.is_bit_set(cursor as usize) {
                return Some((cursor + 1).into());
            }
        }
    }
}

impl IntoIterator for ConstraintList {
    type Item = SudokuNum;
    type IntoIter = ConstraintListIter;

    fn into_iter(self) -> ConstraintListIter {
        ConstraintListIter {
            bits: self.0,
            cursor: 0,
        }
    }
}

impl<'a> IntoIterator for &'a ConstraintList {
    type Item = SudokuNum;
    type IntoIter = ConstraintListIter;

    fn into_iter(self) -> ConstraintListIter {
        ConstraintListIter {
            bits: self.0,
            cursor: 0,
        }
    }
}
impl std::fmt::Debug for ConstraintList {
    fn fmt(&self, f: &mut fmt::Formatter) -> std::fmt::Result {
        let cons = &self.into_iter().map(|n| n as u8).collect::<Vec<u8>>();
        write!(f, "{cons:?}")
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::Free => write!(f, "."),
        }
    }
}

pub fn parse_board(board: Vec<Vec<char>>) -> Board {
    let mut new_board = std::array::from_fn(|_| std::array::from_fn(|_| Cell::Free));

    for row_index in 0..9 {
        for col_index in 0..9 {
            let char_cell = board[row_index][col_index];
            let cell = match char_cell {
                '0'..='9' => Cell::Number(parse_char_to_sudoku_num(char_cell)),
                '.' => Cell::Free,
                _ => panic!("invalid char"),
            };
            new_board[row_index][col_index] = cell;
        }
    }

    Board(new_board)
}

pub fn parse_char_to_sudoku_num(c: char) -> SudokuNum {
    match c {
        '1' => SudokuNum::One,
        '2' => SudokuNum::Two,
        '3' => SudokuNum::Three,
        '4' => SudokuNum::Four,
        '5' => SudokuNum::Five,
        '6' => SudokuNum::Six,
        '7' => SudokuNum::Seven,
        '8' => SudokuNum::Eight,
        '9' => SudokuNum::Nine,
        _ => unreachable!("failed to parse char to sudoku number"),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::assert_matches::assert_matches;

    type BoardIndex = BigBoardIndex;

    #[test]
    fn constraint_list_full() {
        let list = ConstraintList::full();
        assert_eq!(list.0 .0, 0b_0000_0001_1111_1111);
    }

    #[test]
    fn constraint_list_empty() {
        let list = ConstraintList::empty();
        assert_eq!(list.0 .0, 0b_0000_0000_0000_0000);
    }

    #[test]
    fn constraint_list_is_empty() {
        let mut list = ConstraintList::empty();
        assert!(list.is_empty());

        list.insert(SudokuNum::One);
        assert!(!list.is_empty());
    }

    #[test]
    fn naked_singles_full() {
        let list = ConstraintList::full();
        assert_matches!(list.naked_single(), None);
    }

    #[test]
    fn naked_singles_empty() {
        let list = ConstraintList::empty();
        assert_matches!(list.naked_single(), None);
    }

    #[test]
    fn constraint_list_insert() {
        let mut list = ConstraintList::empty();
        list.insert(SudokuNum::One);
        assert_eq!(list.0 .0, 0b_0000_0000_0000_0001);
    }

    #[test]
    fn constraint_list_naked_single_1() {
        let mut list = ConstraintList::empty();
        list.insert(SudokuNum::One);
        assert_matches!(list.naked_single(), Some(SudokuNum::One));
    }

    #[test]
    fn constraint_list_naked_single_2() {
        let mut list = ConstraintList::from_raw_bits(0b_0000_0000_0010_0010);
        list.insert(SudokuNum::One);
        assert_matches!(list.naked_single(), None);
    }

    #[test]
    fn constraint_list_remove() {
        let mut list = ConstraintList::full();
        list.remove(SudokuNum::One);
        assert_eq!(list.0 .0, 0b_0000_0001_1111_1110);
    }

    #[test]
    fn constraint_list_remove_all() {
        let mut list1 = ConstraintList::full();
        let list2 = ConstraintList(U9BitArray::new(0b_0000_0000_0000_1111));
        list1.remove_all(list2);
        assert_eq!(list1.0 .0, 0b_0000_0001_1111_0000);
    }

    #[test]
    fn constraint_list_contains() {
        let mut list = ConstraintList::empty();
        list.insert(SudokuNum::One);
        assert!(list.contains(SudokuNum::One));
        assert!(!list.contains(SudokuNum::Two));
    }

    #[test]
    fn constraint_list_contains_all() {
        let mut list = ConstraintList::empty();
        list.insert(SudokuNum::One);
        assert!(list.contains(SudokuNum::One));
        assert!(!list.contains(SudokuNum::Two));

        let full = ConstraintList::full();
        assert!(!list.contains_all(full));

        let sparse = ConstraintList::from_raw_bits(0b_0000_0001_0010_1110);
        assert!(sparse.contains_all(ConstraintList::from_raw_bits(0b_0000_0001_0010_1110)));

        let subset = ConstraintList::from_raw_bits(0b_0000_0001_0010_1110);
        assert!(!subset.contains_all(ConstraintList::from_raw_bits(0b_0000_0001_1110_1110)));
    }

    #[test]
    fn constraint_list_len() {
        let mut list = ConstraintList::empty();
        assert_eq!(list.len(), 0);

        list.insert(SudokuNum::One);
        assert_eq!(list.len(), 1);

        list.insert(SudokuNum::Two);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn intersection_empty() {
        let c0 = ConstraintList::empty();
        let c1 = ConstraintList::empty();
        let c2 = ConstraintList::empty();

        assert_eq!(ConstraintList::intersection(c0, c1, c2), ConstraintList::empty())
    }

    #[test]
    fn intersection_full() {
        let c0 = ConstraintList::full();
        let c1 = ConstraintList::full();
        let c2 = ConstraintList::full();

        assert_eq!(ConstraintList::intersection(c0, c1, c2), ConstraintList::full())
    }

    #[test]
    fn intersection_1() {
        let c0 = ConstraintList::from_raw_bits(0b_0000_0001_1100_0101);
        let c1 = ConstraintList::from_raw_bits(0b_0000_0000_1000_0101);
        let c2 = ConstraintList::from_raw_bits(0b_0000_0001_1000_0100);

        assert_eq!(
            ConstraintList::intersection(c0, c1, c2),
            ConstraintList::from_raw_bits(0b_0000_0000_1000_0100)
        );
    }

    #[test]
    fn row_iter() {
        let mut iter = RowIter::new(5);
        for i in 0..9 {
            assert_eq!(iter.next(), Some(BoardIndex::new(5, i)));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn col_iter() {
        let mut iter = ColIter::new(5);
        for i in 0..9 {
            assert_eq!(iter.next(), Some(BoardIndex::new(i, 5)));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn square_iter() {
        let mut iter = BoxIter::new(4); // middle square
        let expected = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)];
        for &pos in &expected {
            let index = BoardIndex::new(pos.0, pos.1);
            assert_eq!(iter.next(), Some(index));
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn indices_iter() {
        let mut iter = BoardIter::new();
        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(iter.next(), Some(BoardIndex::new(row, col)));
            }
        }
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn combinations_iter() {
        let cons = ConstraintList::from_raw_bits(0b0000_0000_0000_1011);
        let results = cons.combinations(2).collect::<Vec<ConstraintList>>();

        dbg!(cons);
        assert_eq!(
            results,
            vec![
                ConstraintList::from_raw_bits(0b0000_0000_0000_0011),
                ConstraintList::from_raw_bits(0b0000_0000_0000_1001),
                ConstraintList::from_raw_bits(0b0000_0000_0000_1010),
            ]
        );
    }

    #[test]
    fn combinations_iter_single_bit() {
        let cons = ConstraintList::from_raw_bits(0b0000_0000_0000_1000);
        let results = cons.combinations(1).collect::<Vec<ConstraintList>>();

        dbg!(cons);
        assert_eq!(results, vec![ConstraintList::from_raw_bits(0b0000_0000_0000_1000)]);
    }

    #[test]
    fn combinations_iter_no_bits() {
        let mut iter = ConstraintList::from_raw_bits(0b0000_0000_0000_0000).combinations(1);

        assert_eq!(iter.next(), None);
    }
    #[test]
    fn combinations_iter_k3() {
        let cons = ConstraintList::from_raw_bits(0b0000_0000_0000_1111);
        let results = cons.combinations(3).collect::<Vec<ConstraintList>>();

        dbg!(cons);
        assert_eq!(
            results,
            vec![
                ConstraintList::from_raw_bits(0b0000_0000_0000_0111),
                ConstraintList::from_raw_bits(0b0000_0000_0000_1011),
                ConstraintList::from_raw_bits(0b0000_0000_0000_1101),
                ConstraintList::from_raw_bits(0b0000_0000_0000_1110),
            ]
        );
    }

    #[test]
    fn combinations_iter_k4() {
        let cons = ConstraintList::from_raw_bits(0b0000_0000_0000_1111);
        let results = cons.combinations(4).collect::<Vec<ConstraintList>>();

        dbg!(cons);
        assert_eq!(results, vec![ConstraintList::from_raw_bits(0b0000_0000_0000_1111),]);
    }

    #[test]
    fn combinations_iter_k1_multiple_bits() {
        let cons = ConstraintList::from_raw_bits(0b0000_0000_0000_1111);
        let results = cons.combinations(1).collect::<Vec<ConstraintList>>();

        assert_eq!(
            results,
            vec![
                ConstraintList::from_raw_bits(0b0000_0000_0000_0001),
                ConstraintList::from_raw_bits(0b0000_0000_0000_0010),
                ConstraintList::from_raw_bits(0b0000_0000_0000_0100),
                ConstraintList::from_raw_bits(0b0000_0000_0000_1000),
            ]
        );
    }

    #[test]
    fn combinations_iter_k1_full() {
        let cons = ConstraintList::full();
        let results = cons.combinations(1).collect::<Vec<ConstraintList>>();

        assert_eq!(
            results,
            vec![
                ConstraintList::from_raw_bits(0b0000_0000_0000_0001),
                ConstraintList::from_raw_bits(0b0000_0000_0000_0010),
                ConstraintList::from_raw_bits(0b0000_0000_0000_0100),
                ConstraintList::from_raw_bits(0b0000_0000_0000_1000),
                ConstraintList::from_raw_bits(0b0000_0000_0001_0000),
                ConstraintList::from_raw_bits(0b0000_0000_0010_0000),
                ConstraintList::from_raw_bits(0b0000_0000_0100_0000),
                ConstraintList::from_raw_bits(0b0000_0000_1000_0000),
                ConstraintList::from_raw_bits(0b0000_0001_0000_0000),
            ]
        );
    }

    #[test]
    fn combinations_iter_k1_single_bit_high() {
        let cons = ConstraintList::from_raw_bits(0b0000_0000_1000_0000);
        let results = cons.combinations(1).collect::<Vec<ConstraintList>>();

        assert_eq!(results, vec![ConstraintList::from_raw_bits(0b0000_0000_1000_0000)]);
    }

    #[test]
    fn combinations_iter_k1_no_bits_high() {
        let mut iter = ConstraintList::from_raw_bits(0b0000_0000_0000_0000).combinations(1);

        assert_eq!(iter.next(), None);
    }
    #[test]
    fn combinations_iter_k2_multiple_bits() {
        let cons = ConstraintList::from_raw_bits(0b0000_0000_0001_1111);
        let results = cons.combinations(2).collect::<Vec<ConstraintList>>();

        dbg!(&cons);

        assert_eq!(
            results,
            vec![
                ConstraintList::from_raw_bits(0b0000_0000_0000_0011),
                ConstraintList::from_raw_bits(0b0000_0000_0000_0101),
                ConstraintList::from_raw_bits(0b0000_0000_0000_0110),
                ConstraintList::from_raw_bits(0b0000_0000_0000_1001),
                ConstraintList::from_raw_bits(0b0000_0000_0000_1010),
                ConstraintList::from_raw_bits(0b0000_0000_0000_1100),
                ConstraintList::from_raw_bits(0b0000_0000_0001_0001),
                ConstraintList::from_raw_bits(0b0000_0000_0001_0010),
                ConstraintList::from_raw_bits(0b0000_0000_0001_0100),
                ConstraintList::from_raw_bits(0b0000_0000_0001_1000),
            ]
        );
    }

    #[test]
    fn combinations_iter_k2_single_bit_high() {
        let cons = ConstraintList::from_raw_bits(0b0000_0000_1000_0001);
        let results = cons.combinations(2).collect::<Vec<ConstraintList>>();

        assert_eq!(results, vec![ConstraintList::from_raw_bits(0b0000_0000_1000_0001)]);
    }

    #[test]
    fn combinations_iter_k2_no_bits_high() {
        let mut iter = ConstraintList::from_raw_bits(0b0000_0000_0000_0000).combinations(2);

        assert_eq!(iter.next(), None);
    }

    #[test]
    fn combinations_iter_k2_single_bit() {
        let cons = ConstraintList::from_raw_bits(0b0000_0000_0000_1000);
        let results = cons.combinations(2).collect::<Vec<ConstraintList>>();

        assert_eq!(results, vec![]);
    }

    #[test]
    fn constraint_list_iter() {
        let it = ConstraintList::from_raw_bits(0b0000_0000_1001_1000).into_iter();
        let results = it.collect::<Vec<SudokuNum>>();

        assert_eq!(results, vec![SudokuNum::Four, SudokuNum::Five, SudokuNum::Eight]);
    }
    #[test]
    fn board_not_solved_empty_cells() {
        let codegolf = parse_board(vec![
            vec!['.', '.', '.', '7', '.', '.', '.', '.', '.'],
            vec!['1', '.', '.', '.', '.', '.', '.', '.', '.'],
            vec!['.', '.', '.', '4', '3', '.', '2', '.', '.'],
            vec!['.', '.', '.', '.', '.', '.', '.', '.', '6'],
            vec!['.', '.', '.', '5', '.', '9', '.', '.', '.'],
            vec!['.', '.', '.', '.', '.', '.', '4', '1', '8'],
            vec!['.', '.', '.', '.', '8', '1', '.', '.', '.'],
            vec!['.', '.', '2', '.', '.', '.', '.', '5', '.'],
            vec!['.', '4', '.', '.', '.', '.', '3', '.', '.'],
        ]);
        assert_eq!(codegolf.is_solved(), false);
    }
    #[test]
    fn board_not_solved_1() {
        let board = parse_board(vec![
            vec!['9', '3', '4', '6', '7', '8', '9', '1', '2'],
            vec!['6', '7', '2', '1', '9', '5', '3', '4', '8'],
            vec!['1', '9', '8', '3', '4', '2', '5', '6', '7'],
            vec!['8', '5', '9', '7', '6', '1', '4', '2', '3'],
            vec!['4', '2', '6', '8', '5', '3', '7', '9', '1'],
            vec!['7', '1', '3', '9', '2', '4', '8', '5', '6'],
            vec!['9', '6', '1', '5', '3', '7', '2', '8', '4'],
            vec!['2', '8', '7', '4', '1', '9', '6', '3', '5'],
            vec!['3', '4', '5', '2', '8', '6', '1', '7', '5'],
        ]);
        assert_eq!(board.is_solved(), false);
    }

    #[test]
    fn board_is_solved_2() {
        let board = parse_board(vec![
            vec!['5', '3', '4', '6', '7', '8', '9', '1', '2'],
            vec!['6', '7', '2', '1', '9', '5', '3', '4', '8'],
            vec!['1', '9', '8', '3', '4', '2', '5', '6', '7'],
            vec!['8', '5', '9', '7', '6', '1', '4', '2', '3'],
            vec!['4', '2', '6', '8', '5', '3', '7', '9', '1'],
            vec!['7', '1', '3', '9', '2', '4', '8', '5', '6'],
            vec!['9', '6', '1', '5', '3', '7', '2', '8', '4'],
            vec!['2', '8', '7', '4', '1', '9', '6', '3', '5'],
            vec!['3', '4', '5', '2', '8', '6', '1', '7', '9'],
        ]);
        assert_eq!(board.is_solved(), true);
    }
}
