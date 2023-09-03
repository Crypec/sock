#![feature(let_chains)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(internal_features)]
#![feature(rustc_attrs)]
#![feature(associated_type_bounds)]
#![feature(assert_matches)]
#![feature(test)]
//#![allow(dead_code)]

// #![warn(clippy::restriction)]

use crate::board::{parse_char_to_sudoku_num, Board, Cell};
use crate::solver::Solver;
use rayon::prelude::*;
use std::assert_matches::assert_matches;

mod board;
mod solver;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

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

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let test_data = std::fs::read_to_string("test_data.txt").unwrap();
    let boards = parse_boards_list(&test_data);

    let now = std::time::Instant::now();

    #[cfg(not(feature = "no-jobs"))]
    {
        boards.into_par_iter().for_each(|board| {
            let solver = Solver::new(board);
            let res = solver.solve();
            assert_matches!(res, Ok(b) if b.is_solved());
        });
    }

    #[cfg(feature = "no-jobs")]
    {
        for (index, board) in boards.into_iter().enumerate() {
            let now = std::time::Instant::now();
            let solver = Solver::new(board);
            let res = solver.solve();

            println!("solved: {index} :: {:?}", now.elapsed());

            assert_matches!(res, Ok(b) if b.is_solved());
        }
    }

    println!("took :: {:?}", now.elapsed());
    println!("size of Solver: {} bytes", std::mem::size_of::<Solver>());
}
