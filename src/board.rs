#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
// #![warn(clippy::restriction)]

use cli_table::{Style, Table};
use std::fmt;

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct Board(pub [[Cell; 9]; 9]);

const MIDDLE_OF_SQUARE_INDEXES: [(i8, i8); 9] =
    [(1, 1), (1, 4), (1, 7), (4, 1), (4, 4), (4, 7), (7, 1), (7, 4), (7, 7)];

const OFFSETS: [(i8, i8); 9] = [
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    fn eq(&self, other: &u8) -> bool {
        (*self as u8) == *other
    }
}

impl std::fmt::Display for SudokuNum {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", *self as u8)
    }
}

impl From<usize> for SudokuNum {
    fn from(number: usize) -> SudokuNum {
        match number {
            1 => SudokuNum::One,
            2 => SudokuNum::Two,
            3 => SudokuNum::Three,
            4 => SudokuNum::Four,
            5 => SudokuNum::Five,
            6 => SudokuNum::Six,
            7 => SudokuNum::Seven,
            8 => SudokuNum::Eight,
            9 => SudokuNum::Nine,
            _ => panic!("failed to convert `{number}` to suduku number!"),
        }
    }
}

impl From<u8> for SudokuNum {
    fn from(number: u8) -> SudokuNum {
        (number as usize).into()
    }
}

#[derive(Hash, Clone, Eq, PartialEq)]
pub struct ConstraintList(pub U16BitArray);

impl ConstraintList {
    pub fn full() -> Self {
        ConstraintList(U16BitArray(0b_0000_0001_1111_1111))
    }

    pub fn empty() -> Self {
        ConstraintList(U16BitArray(0b_0000_0000_0000_0000))
    }

    pub fn is_empty(&self) -> bool {
        self.0.count_ones() == 0
    }

    pub fn insert(&mut self, num: SudokuNum) {
        self.0.set_bit((num as u8) - 1);
    }

    pub fn remove(&mut self, num: SudokuNum) {
        self.0.clear_bit((num as u8) - 1);
    }

    pub fn contains(&self, needle: SudokuNum) -> bool {
        self.0.is_bit_set((needle as u8) - 1)
    }

    pub fn first(&self) -> Option<SudokuNum> {
        self.0.first_index().map(|x| (x + 1).into())
    }
    pub fn len(&self) -> u32 {
        self.0.count_ones()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct U16BitArray(u16);

impl U16BitArray {
    pub fn set_bit(&mut self, index: u8) {
        assert!(index <= 15, "index out of range");
        self.0 |= 1 << index;
    }

    pub fn is_bit_set(&self, index: u8) -> bool {
        assert!(index <= 15, "index out of range");
        self.0 & (1 << index) != 0
    }

    pub fn clear_bit(&mut self, index: u8) {
        assert!(index <= 15, "index out of range");
        self.0 &= !(1 << index);
    }

    pub fn count_ones(&self) -> u32 {
        self.0.count_ones()
    }
    pub fn first_index(&self) -> Option<u8> {
        if self.0 == 0 {
            return None;
        }
        for i in 0..=15 {
            if self.is_bit_set(i) {
                return Some(i);
            }
        }
        None
    }
}

pub struct ConstraintListIter {
    bits: U16BitArray,
    cursor: u8,
}

impl Iterator for ConstraintListIter {
    type Item = SudokuNum;
    fn next(&mut self) -> Option<Self::Item> {
        None
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
        write!(f, "{:?}", cons)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Cell {
    Number(SudokuNum),
    Constrained(ConstraintList),
    Free,
}

// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
// pub enum Cell {
//     Number(u8),
//     Constrained(Vec<u8>),
//     Free,
// }

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::Constrained(c) => write!(f, "{c:?}"),
            Self::Free => write!(f, "."),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct BoardNotSolvableError;

impl fmt::Display for BoardNotSolvableError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "failed to solve board")
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

// fn to_solution(board: Board) -> Vec<Vec<char>> {
//     let mut new_board = vec![];
//     for row in board.0 {
//         let mut new_row = vec![];
//         for cell in row {
//             let ch = match cell {
//                 Cell::Number(n) => n as char,
//                 Cell::Free => '.',
//                 Cell::Constrained(_) => panic!(),
//             };
//             new_row.push(ch);
//         }
//         new_board.push(new_row);
//     }
//     new_board
// }

pub fn print_board(board: &Board) {
    let table = board.0.clone().table().bold(true).display().unwrap();

    println!(
        "\n{table}\n",
        // termion::clear::All,
        // termion::cursor::Goto(1, 1)
    );
}

impl Board {
    pub fn is_solved(&self) -> bool {
        for row in &self.0 {
            for cell in row {
                if let Cell::Constrained(_) | Cell::Free = cell {
                    return false;
                }
            }
        }

        // check horizontal rows
        for row in &self.0 {
            let mut required_nums = ConstraintList::full();
            for cell in row {
                if let Cell::Number(n) = cell {
                    required_nums.remove(*n)
                }
            }
            if !required_nums.is_empty() {
                return false;
            }
        }

        // check vertical columns
        for col_index in 0..9 {
            let mut required_nums = ConstraintList::full();
            for row_index in 0..9 {
                let cell = &self.0[row_index][col_index];
                if let Cell::Number(n) = cell {
                    required_nums.remove(*n)
                }
            }
            if !required_nums.is_empty() {
                return false;
            }
        }

        for (square_row_index, square_col_index) in MIDDLE_OF_SQUARE_INDEXES {
            let mut required_nums = ConstraintList::full();
            for (offset_y, offset_x) in OFFSETS {
                let row_index = (square_row_index + offset_y) as usize;
                let col_index = (square_col_index + offset_x) as usize;

                let cell = &self.0[row_index][col_index];
                if let Cell::Number(n) = cell {
                    required_nums.remove(*n);
                }
            }
            if !required_nums.is_empty() {
                return false;
            }
        }

        true
    }
}
