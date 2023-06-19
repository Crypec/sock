#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
// #![warn(clippy::restriction)]

use cli_table::{Style, Table};
use rustc_hash::{FxHashMap, FxHasher};
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct Board(pub [[Cell; 9]; 9]);

// impl IntoIterator for Board {
//     type Item = [Cell; 9];
//     type IntoIter = std::array::IntoIter<[Cell; 9], 9>;
//     fn into_iter(self) -> Self::IntoIter {
//         self.0.into_iter()
//     }
// }

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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NewCell {
    Number(SudokuNum),
    Constrained([Option<SudokuNum>; 9]),
    Free,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Cell {
    Number(u8),
    Constrained(Vec<u8>),
    Free,
}

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
                '0'..='9' => Cell::Number(char_cell.to_digit(10).unwrap() as u8),
                '.' => Cell::Free,
                _ => panic!("invalid char"),
            };
            new_board[row_index][col_index] = cell;
        }
    }

    Board(new_board)
}

fn to_solution(board: Board) -> Vec<Vec<char>> {
    let mut new_board = vec![];
    for row in board.0 {
        let mut new_row = vec![];
        for cell in row {
            let ch = match cell {
                Cell::Number(n) => n as char,
                Cell::Free => '.',
                Cell::Constrained(_) => panic!(),
            };
            new_row.push(ch);
        }
        new_board.push(new_row);
    }
    new_board
}

fn print_board(board: &Board) {
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
            let mut required_nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
            for cell in row {
                if let Cell::Number(n) = cell {
                    if let Some(index) = required_nums.iter().position(|x| x == n) {
                        required_nums.remove(index);
                    }
                }
            }
            if !required_nums.is_empty() {
                return false;
            }
        }

        // check vertical columns
        for col_index in 0..9 {
            let mut required_nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
            for row_index in 0..9 {
                let cell = &self.0[row_index][col_index];
                if let Cell::Number(n) = cell {
                    if let Some(index) = required_nums.iter().position(|x| x == n) {
                        required_nums.remove(index);
                    }
                }
            }
            if !required_nums.is_empty() {
                return false;
            }
        }

        for (square_row_index, square_col_index) in MIDDLE_OF_SQUARE_INDEXES {
            let mut required_nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
            for (offset_y, offset_x) in OFFSETS {
                let row_index = (square_row_index + offset_y) as usize;
                let col_index = (square_col_index + offset_x) as usize;

                let cell = &self.0[row_index][col_index];
                if let Cell::Number(n) = cell {
                    if let Some(index) = required_nums.iter().position(|x| x == n) {
                        required_nums.remove(index);
                    }
                }
            }
            if !required_nums.is_empty() {
                return false;
            }
        }

        true
    }
}
