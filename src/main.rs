#![feature(let_chains)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![feature(associated_type_bounds)]
#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

// #![warn(clippy::restriction)]

use crate::board::{parse_board, parse_char_to_sudoku_num, Board, Cell};
use crate::solver::{BoardNotSolvableError, Solver};

mod board;
mod solver;

fn parse_boards_list(raw: &str) -> Vec<Board> {
    raw.lines().skip(1).map(parse_board_from_line).collect()
}

fn parse_board_from_line(line: &str) -> Board {
    debug_assert_eq!(line.len(), 81);
    let mut new_board = std::array::from_fn(|_| std::array::from_fn(|_| Cell::Free));
    for (row_index, row) in new_board.iter_mut().enumerate() {
        for (col_index, cell) in row.iter_mut().enumerate() {
            let char_cell = (line.as_bytes()[(row_index * 9) + col_index]) as char;
            let new_cell = match char_cell {
                '1'..='9' => Cell::Number(parse_char_to_sudoku_num(char_cell)),
                '.' | '0' => Cell::Free,
                _ => panic!("invalid char"),
            };
            *cell = new_cell;
        }
    }
    board::Board(new_board)
}

struct GPTSolver {
    grid: [[u8; 9]; 9],
    rows: [u16; 9],
    cols: [u16; 9],
    boxes: [u16; 9],
}

impl GPTSolver {
    fn new(input: [[char; 9]; 9]) -> GPTSolver {
        let mut solver = GPTSolver {
            grid: [[0; 9]; 9],
            rows: [0; 9],
            cols: [0; 9],
            boxes: [0; 9],
        };

        for i in 0..9 {
            for j in 0..9 {
                if input[i][j] != '.' {
                    let num = input[i][j].to_digit(10).unwrap();
                    let bit = 1 << num;
                    solver.grid[i][j] = num as u8;
                    solver.rows[i] |= bit;
                    solver.cols[j] |= bit;
                    solver.boxes[(i / 3) * 3 + j / 3] |= bit;
                }
            }
        }

        solver
    }
    fn solve(&mut self) -> bool {
        for i in 0..9 {
            for j in 0..9 {
                if self.grid[i][j] == 0 {
                    for num in 1..=9 {
                        let bit = 1 << num;
                        if self.rows[i] & bit == 0
                            && self.cols[j] & bit == 0
                            && self.boxes[(i / 3) * 3 + j / 3] & bit == 0
                        {
                            self.grid[i][j] = num as u8;
                            self.rows[i] |= bit;
                            self.cols[j] |= bit;
                            self.boxes[(i / 3) * 3 + j / 3] |= bit;

                            if self.solve() {
                                return true;
                            }

                            self.grid[i][j] = 0;
                            self.rows[i] &= !bit;
                            self.cols[j] &= !bit;
                            self.boxes[(i / 3) * 3 + j / 3] &= !bit;
                        }
                    }
                    return false;
                }
            }
        }
        true
    }
}

use std::fs;

fn parse_gpt_boards(filename: &str) -> Result<Vec<[[char; 9]; 9]>, std::io::Error> {
    let content = fs::read_to_string(filename)?;
    let lines = content.lines().skip(1);

    let mut boards = Vec::new();

    for line in lines {
        let mut board = [['.'; 9]; 9];
        for (i, cell) in line.chars().enumerate() {
            let row = i / 9;
            let col = i % 9;
            board[row][col] = if cell == '0' { '.' } else { cell };
        }
        boards.push(board);
    }

    Ok(boards)
}

fn main() -> Result<(), BoardNotSolvableError> {
    let mut _really_hard_test = parse_board(vec![
        vec!['.', '.', '.', '.', '7', '.', '1', '.', '.'],
        vec!['.', '.', '.', '5', '6', '.', '.', '.', '.'],
        vec!['.', '8', '.', '.', '2', '.', '.', '3', '.'],
        vec!['.', '.', '.', '.', '.', '.', '.', '4', '9'],
        vec!['.', '4', '.', '2', '5', '.', '.', '.', '8'],
        vec!['5', '.', '.', '9', '.', '.', '.', '.', '6'],
        vec!['4', '6', '.', '.', '.', '.', '2', '8', '.'],
        vec!['2', '.', '.', '.', '.', '.', '.', '.', '.'],
        vec!['7', '.', '.', '1', '9', '.', '8', '.', '.'],
    ]);

    let mut _codegolf = parse_board(vec![
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

    // let mut hard_leetcode = parse_board(vec![
    //     vec!['.', '.', '.', '.', '.', '7', '.', '.', '9'],
    //     vec!['.', '4', '.', '.', '8', '1', '2', '.', '.'],
    //     vec!['.', '.', '.', '9', '.', '.', '.', '1', '.'],
    //     vec!['.', '.', '5', '3', '.', '.', '.', '7', '2'],
    //     vec!['2', '9', '3', '.', '.', '.', '.', '5', '.'],
    //     vec!['.', '.', '.', '.', '.', '5', '3', '.', '.'],
    //     vec!['8', '.', '.', '.', '2', '3', '.', '.', '.'],
    //     vec!['7', '.', '.', '.', '5', '.', '.', '4', '.'],
    //     vec!['5', '3', '1', '.', '7', '.', '.', '.', '.'],
    // ]);

    let mut _test_board = parse_board(vec![
        vec!['5', '3', '.', '.', '7', '.', '.', '.', '.'],
        vec!['6', '.', '.', '1', '9', '5', '.', '.', '.'],
        vec!['.', '9', '8', '.', '.', '.', '.', '6', '.'],
        vec!['8', '.', '.', '.', '6', '.', '.', '.', '3'],
        vec!['4', '.', '.', '8', '.', '3', '.', '.', '1'],
        vec!['7', '.', '.', '.', '2', '.', '.', '.', '6'],
        vec!['.', '6', '.', '.', '.', '.', '2', '8', '.'],
        vec!['.', '.', '.', '4', '1', '9', '.', '.', '5'],
        vec!['.', '.', '.', '.', '8', '.', '.', '7', '9'],
    ]);

    let _test_board_solution = parse_board(vec![
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

    let test_data = std::fs::read_to_string("test_data.txt").unwrap();
    let boards = parse_boards_list(&test_data);
    // let boards = parse_gpt_boards("test_data.txt").unwrap();

    // let now = std::time::Instant::now();
    // for (index, board) in boards.into_iter().enumerate() {
    //     let mut solver = Solver::new(board.clone());
    //     let now = std::time::Instant::now();
    //     println!("{index}");
    //     let _ = solver.solve()?;
    //     println!("{index} :: in {:?}", now.elapsed());
    // }

    // println!("took :: {:?}", now.elapsed());

    let _board_4_solution = parse_board(vec![
        vec!['3', '4', '6', '7', '9', '5', '8', '1', '2'],
        vec!['2', '5', '8', '4', '3', '1', '6', '9', '7'],
        vec!['9', '7', '1', '8', '6', '2', '5', '4', '3'],

        vec!['1', '2', '9', '5', '7', '6', '4', '3', '8'],
        vec!['8', '3', '5', '2', '1', '4', '7', '6', '9'],
        vec!['7', '6', '4', '3', '8', '9', '2', '5', '1'],

        vec!['5', '1', '7', '9', '4', '8', '3', '2', '6'],
        vec!['4', '9', '3', '6', '2', '7', '1', '8', '5'],
        vec!['6', '8', '2', '1', '5', '3', '9', '7', '4'],
    ]);

    let board = &boards[4];
    dbg!(&board);
    let mut solver = Solver::new(boards[4].clone());
    let now = std::time::Instant::now();
    let status = solver.solve();
    dbg!(&solver.board);
    dbg!(&solver.is_subset(&_board_4_solution));
    println!("status :: {status:?} in {:?}", now.elapsed());

    // print_board(&mut boards[4]);
    Ok(())
}

// #[cfg(test)]
// mod test {
//     use crate::*;

//     #[test]
//     fn board_not_solved_empty_cells() {
//         let codegolf = parse_board(vec![
//             vec!['.', '.', '.', '7', '.', '.', '.', '.', '.'],
//             vec!['1', '.', '.', '.', '.', '.', '.', '.', '.'],
//             vec!['.', '.', '.', '4', '3', '.', '2', '.', '.'],
//             vec!['.', '.', '.', '.', '.', '.', '.', '.', '6'],
//             vec!['.', '.', '.', '5', '.', '9', '.', '.', '.'],
//             vec!['.', '.', '.', '.', '.', '.', '4', '1', '8'],
//             vec!['.', '.', '.', '.', '8', '1', '.', '.', '.'],
//             vec!['.', '.', '2', '.', '.', '.', '.', '5', '.'],
//             vec!['.', '4', '.', '.', '.', '.', '3', '.', '.'],
//         ]);
//         assert_eq!(board_is_solved(&codegolf), false);
//     }
//     #[test]
//     fn board_not_solved_1() {
//         let board = parse_board(vec![
//             vec!['9', '3', '4', '6', '7', '8', '9', '1', '2'],
//             vec!['6', '7', '2', '1', '9', '5', '3', '4', '8'],
//             vec!['1', '9', '8', '3', '4', '2', '5', '6', '7'],
//             vec!['8', '5', '9', '7', '6', '1', '4', '2', '3'],
//             vec!['4', '2', '6', '8', '5', '3', '7', '9', '1'],
//             vec!['7', '1', '3', '9', '2', '4', '8', '5', '6'],
//             vec!['9', '6', '1', '5', '3', '7', '2', '8', '4'],
//             vec!['2', '8', '7', '4', '1', '9', '6', '3', '5'],
//             vec!['3', '4', '5', '2', '8', '6', '1', '7', '5'],
//         ]);
//         assert_eq!(board_is_solved(&board), false);
//     }

//     #[test]
//     fn board_is_solved_2() {
//         let board = parse_board(vec![
//             vec!['5', '3', '4', '6', '7', '8', '9', '1', '2'],
//             vec!['6', '7', '2', '1', '9', '5', '3', '4', '8'],
//             vec!['1', '9', '8', '3', '4', '2', '5', '6', '7'],
//             vec!['8', '5', '9', '7', '6', '1', '4', '2', '3'],
//             vec!['4', '2', '6', '8', '5', '3', '7', '9', '1'],
//             vec!['7', '1', '3', '9', '2', '4', '8', '5', '6'],
//             vec!['9', '6', '1', '5', '3', '7', '2', '8', '4'],
//             vec!['2', '8', '7', '4', '1', '9', '6', '3', '5'],
//             vec!['3', '4', '5', '2', '8', '6', '1', '7', '9'],
//         ]);
//         assert_eq!(board_is_solved(&board), true);
//     }
// }
