#![feature(let_chains)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(internal_features)]
#![feature(rustc_attrs)]
#![feature(associated_type_bounds)]
#![feature(assert_matches)]
#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

// #![warn(clippy::restriction)]

use crate::board::{parse_char_to_sudoku_num, Board, Cell};
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

fn main() -> Result<(), BoardNotSolvableError> {
    let test_data = std::fs::read_to_string("test_data.txt").unwrap();
    let boards = parse_boards_list(&test_data);

    let now = std::time::Instant::now();
    for i in 0..=100 {
        let board = boards[i].clone();
        let mut solver = Solver::new(board.clone());
        let now = std::time::Instant::now();
        println!("{i}");
        let status = solver.solve();
        dbg!(&solver.board);
        println!("{i} :: in {:?}", now.elapsed());
        assert!(status.is_ok());
    }

    println!("took :: {:?}", now.elapsed());

    Ok(())
}
