#![warn(clippy::nursery)]
// #![warn(clippy::pedantic)]
#![allow(internal_features)]
#![feature(rustc_attrs)]
#![feature(associated_type_bounds)]
#![feature(assert_matches)]
#![feature(test)]
#![allow(dead_code)]
#![allow(clippy::inline_always)]
#![feature(maybe_uninit_uninit_array)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// #![warn(clippy::restriction)]

use crate::board::{parse_board_from_line, Board};
use crate::solver::Solver;

#[cfg(not(feature = "no-jobs"))]
use rayon::prelude::*;

use std::assert_matches::assert_matches;

mod board;
mod generated_lut;
mod solver;
mod subset_cache;

mod visualize;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn parse_boards_list(raw: &str) -> Vec<Board> {
    raw.lines().skip(1).map(parse_board_from_line).collect()
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let test_data = std::fs::read_to_string("test_data/codegolf").unwrap();
    let boards = parse_boards_list(&test_data);

    let now = std::time::Instant::now();

    #[cfg(not(feature = "no-jobs"))]
    {
        boards.into_par_iter().for_each(|board| {
            let mut solver = Solver::new(board);
            let res = solver.solve();

            // solver.visualize();

            assert_matches!(res, Ok(b) if b.is_solved());
        });
    }

    #[cfg(feature = "no-jobs")]
    {
        for (index, board) in boards.into_iter().enumerate() {
            let now = std::time::Instant::now();
            let mut solver = Solver::new(board);
            let res = solver.solve();

            assert_matches!(res, Ok(b) if b.is_solved());

            println!("solved: {index} :: {:?}", now.elapsed());
        }
    }

    println!("took :: {:?}", now.elapsed());
    println!("size of Solver: {} bytes", std::mem::size_of::<Solver>());
}
