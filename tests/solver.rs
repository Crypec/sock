#![feature(assert_matches)]

use std::assert_matches::assert_matches;
use std::path::Path;

#[cfg(not(feature = "no-jobs"))]
use rayon::prelude::*;

use sock::board::parse_board_from_line;
use sock::solver::*;

fn check_against_solution<P: AsRef<Path>>(path: P) -> std::io::Result<()> {
    let test_data = std::fs::read_to_string(path).expect("failed to read codegolf test data");

    #[cfg(feature = "no-jobs")]
    for line in test_data.lines() {
        if line.starts_with('#') {
            continue;
        }

        let unsolved = parse_board_from_line(&line[..81]);
        let solved = parse_board_from_line(&line[84..]);

        let mut solver = Solver::new(unsolved);
        let res = solver.solve();

        assert_matches!(res, Ok(board) if board.is_solved());
        assert_eq!(res.unwrap(), solved)
    }

    #[cfg(not(feature = "no-jobs"))]
    {
        let boards = test_data
            .lines()
            .filter_map(|line| {
                if line.starts_with('#') {
                    None
                } else {
                    let unsolved = parse_board_from_line(&line[..81]);
                    let solved = parse_board_from_line(&line[84..]);

                    Some((unsolved, solved))
                }
            })
            .collect::<Vec<_>>();

        boards.into_par_iter().for_each(|(unsolved, solved)| {
            let mut solver = Solver::new(unsolved);
            let res = solver.solve();

            assert_matches!(res, Ok(board) if board.is_solved());
            assert_eq!(res.unwrap(), solved)
        })
    }

    Ok(())
}

#[test]
fn codegolf_dataset() {
    check_against_solution("test_data/codegolf_solved").expect("failed to read solved `codegolf` test data");
}

#[test]
fn kaggle_dataset() {
    check_against_solution("test_data/puzzles0_kaggle_solved").expect("failed to read solved `kaggle` test data");
}

#[test]
fn unbiased_dataset() {
    check_against_solution("test_data/puzzles1_unbiased_solved").expect("failed to read solved `unbiased` test data");
}

#[test]
fn _17_clue_dataset() {
    check_against_solution("test_data/puzzles2_17_clue_solved").expect("failed to read solved `17 clue` test data");
}

#[test]
fn magictour_top1465_dataset() {
    check_against_solution("test_data/puzzles3_magictour_top1465_solved")
        .expect("failed to read solved `magictour 1465` test data");
}

#[test]
fn forum_hardest_1905_dataset() {
    check_against_solution("test_data/puzzles4_forum_hardest_1905_solved")
        .expect("failed to read solved `forum hardest 1905` test data");
}

#[test]
fn forum_hardest_1106_dataset() {
    check_against_solution("test_data/puzzles6_forum_hardest_1106_solved")
        .expect("failed to read solved `forum hardest 1106` test data");
}

// #[test]
// fn serg_benchmark() {
//     check_against_solution("test_data/puzzles7_serg_benchmark_solved")
//         .expect("failed to read solved `serg benchmark` test data");
// }

// #[test]
// fn gen_puzzles() {
//     check_against_solution("test_data/puzzles8_gen_puzzles_solved")
//         .expect("failed to read solved `gen puzzles` test data");
// }
