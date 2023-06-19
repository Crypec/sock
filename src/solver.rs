use rustc_hash::{FxHashMap, FxHasher};
use std::hash::{Hash, Hasher};

use crate::board::*;

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

#[derive(Debug, Copy, Clone)]
pub struct BoardNotSolvableError;

#[derive(Debug, Copy, Clone)]
pub enum SolveStatusProgress {
    Solved,
    MadeProgress,
    NotSolvable,
    Stalling,
}

pub struct Solver {
    board: Board,
    queue: Vec<(u8, u8)>,

    row_missing: [Vec<u8>; 9],
    col_missing: [Vec<u8>; 9],
    square_missing: [Vec<u8>; 9],
}

impl Solver {
    pub fn new(board: Board) -> Self {
        Self {
            board,
            queue: Vec::with_capacity(81),
        }
    }

    fn insert_and_forward_propagate(&mut self, cell: u8, row_index: u8, col_index: u8) {
        self.insert_and_forward_propagate(cell, row_index, col_index);
        while let Some((row_index, col_index)) = self.queue.pop() {
            self.forward_check_simple(row_index, col_index);
        }
    }

    fn insert_initial_constraints(&mut self) {
        for row_index in 0..9 {
            let mut possible_nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
            for col_index in 0..9 {
                let cell = &self.board.0[row_index][col_index];
                if let Cell::Number(n) = &cell {
                    if let Some(index) = possible_nums.iter().position(|x| *x == *n) {
                        possible_nums.remove(index);
                    }
                }
            }
            for col_index in 0..9 {
                let cell = &mut self.board.0[row_index][col_index];
                if let Cell::Free = &cell {
                    *cell = Cell::Constrained(possible_nums.clone());
                }
            }
        }
    }

    // propagate constraints
    fn partially_propagate_constraints(&mut self) {
        fn partially_propagate_row_constraints(board: &mut Board) {
            for row_index in 0..9 {
                let mut found_nums = Vec::with_capacity(9);
                for col_index in 0..9 {
                    let cell = &board.0[row_index][col_index];
                    if let Cell::Number(n) = cell {
                        found_nums.push(*n);
                    }
                }
                for col_index in 0..9 {
                    let cell = &mut board.0[row_index][col_index];
                    if let Cell::Constrained(cons) = cell {
                        for num in &found_nums {
                            if let Some(index) = cons.iter().position(|x| *x == *num) {
                                cons.remove(index);
                            }
                        }
                    }
                }
            }
        }
        fn partially_propagate_col_constraints(board: &mut Board) {
            for col_index in 0..9 {
                let mut found_nums = Vec::with_capacity(9);
                for row_index in 0..9 {
                    if let Cell::Number(n) = board.0[row_index][col_index] {
                        found_nums.push(n);
                    }
                }
                for row_index in 0..9 {
                    let cell = &mut board.0[row_index][col_index];
                    if let Cell::Constrained(cons) = cell {
                        for num in &found_nums {
                            if let Some(index) = cons.iter().position(|x| *x == *num) {
                                cons.remove(index);
                            }
                        }
                    }
                }
            }
        }
        fn partially_propagate_square_constraints(board: &mut Board) {
            for (square_row_index, square_col_index) in MIDDLE_OF_SQUARE_INDEXES {
                let mut found_nums = Vec::with_capacity(9);
                for (offset_y, offset_x) in OFFSETS {
                    let row_index = (square_row_index + offset_y) as usize;
                    let col_index = (square_col_index + offset_x) as usize;
                    if let Cell::Number(n) = board.0[row_index][col_index] {
                        found_nums.push(n);
                    }
                }

                for (offset_y, offset_x) in OFFSETS {
                    let row_index = (square_row_index + offset_y) as usize;
                    let col_index = (square_col_index + offset_x) as usize;
                    if let Cell::Constrained(ref mut constraints) = &mut board.0[row_index][col_index] {
                        for num in &found_nums {
                            if let Some(index) = constraints.iter().position(|x| x == num) {
                                constraints.remove(index);
                            }
                        }
                    }
                }
            }
        }
        partially_propagate_row_constraints(&mut self.board);
        partially_propagate_col_constraints(&mut self.board);

        partially_propagate_square_constraints(&mut self.board);
    }

    pub fn solve(&mut self) -> SolveStatusProgress {
        self.solve_internal(true)
    }

    fn solve_board_dfs(&mut self) -> SolveStatusProgress {
        for row_index in 0..9 {
            for col_index in 0..9 {
                let cell = self.board.0[row_index][col_index].clone();
                if let Cell::Constrained(cons) = cell {
                    for c in cons.clone() {
                        self.board.0[row_index][col_index] = Cell::Number(c);
                        // print_board(&new_board);
                        if let SolveStatusProgress::Solved = self.solve_internal(false) {
                            return SolveStatusProgress::Solved;
                        }
                        self.board.0[row_index][col_index] = Cell::Constrained(cons.clone());
                    }
                    return SolveStatusProgress::NotSolvable;
                }
            }
        }
        SolveStatusProgress::NotSolvable
    }

    fn solve_internal(&mut self, is_first_iteration: bool) -> SolveStatusProgress {
        if is_first_iteration {
            self.insert_initial_constraints();
        }
        while !self.board.is_solved() {
            self.partially_propagate_constraints();
            let status = self.insert_forced_constraints();
            match status {
                SolveStatusProgress::Stalling => return self.solve_board_dfs(),
                SolveStatusProgress::NotSolvable => return SolveStatusProgress::NotSolvable,
                SolveStatusProgress::Solved => return SolveStatusProgress::Solved,
                SolveStatusProgress::MadeProgress => {}
            };
            // let time_out = std::time::Duration::from_secs(1);
            // std::thread::sleep(time_out);
            // print_board(&board);
        }
        SolveStatusProgress::Solved
    }

    fn insert_forced_constraints(&mut self) -> SolveStatusProgress {
        fn insert_obviously_forced_constraints(board: &mut Board) -> (SolveStatusProgress, bool) {
            let mut made_progress = false;

            // a `forced` constraint is a constraint with len == 1
            for row in board.0.iter_mut() {
                for cell in row {
                    match cell {
                        // puzzle not solvable
                        Cell::Constrained(cons) if cons.is_empty() => {
                            // print_board(board);
                            return (SolveStatusProgress::NotSolvable, made_progress);
                        }
                        Cell::Constrained(cons) if cons.len() == 1 => {
                            *cell = Cell::Number(cons.last().cloned().unwrap());
                            made_progress = true;
                        }
                        _ => continue,
                    }
                }
            }
            // this seems to be wrong because we could be stalling after ending the loop without knowing it
            (SolveStatusProgress::MadeProgress, made_progress)
        }

        fn insert_forced_constraints_in_col(board: &mut Board) -> bool {
            let mut made_progress = false;
            for col_index in 0..9 {
                // occurences number of occurence and indexes
                let mut occurrences: FxHashMap<u8, Vec<(usize, usize)>> = FxHashMap::default();
                for row_index in 0..9 {
                    let cell = &board.0[row_index][col_index];
                    match cell {
                        Cell::Number(n) => {
                            occurrences
                                .entry(*n)
                                .or_insert(Vec::with_capacity(9))
                                .push((row_index, col_index));
                        }
                        Cell::Constrained(cons) => {
                            for c in cons {
                                occurrences
                                    .entry(*c)
                                    .or_insert(Vec::with_capacity(9))
                                    .push((row_index, col_index));
                            }
                        }
                        _ => {}
                    };
                }
                for i in 1..=9 {
                    if let Some(indexes) = occurrences.get(&i) {
                        if indexes.len() == 1 {
                            let (row_index, col_index) = indexes.last().unwrap();
                            board.0[*row_index][*col_index] = Cell::Number(i);
                            made_progress = true;
                        }
                    }
                }
            }
            made_progress
        }

        fn insert_forced_constraints_in_row(board: &mut Board) -> bool {
            let mut made_progress = false;
            for col_index in 0..9 {
                // occurences number of occurence and indexes
                let mut occurrences: FxHashMap<u8, Vec<(usize, usize)>> = FxHashMap::default();
                for row_index in 0..9 {
                    let cell = &board.0[row_index][col_index];
                    match cell {
                        Cell::Number(n) => {
                            occurrences
                                .entry(*n)
                                .or_insert(Vec::with_capacity(9))
                                .push((row_index, col_index));
                        }
                        Cell::Constrained(cons) => {
                            for c in cons {
                                occurrences
                                    .entry(*c)
                                    .or_insert(Vec::with_capacity(9))
                                    .push((row_index, col_index));
                            }
                        }
                        _ => {}
                    };
                }
                for i in 1..=9 {
                    if let Some(indexes) = occurrences.get(&i) {
                        if indexes.len() == 1 {
                            let (row_index, col_index) = indexes.last().unwrap();
                            board.0[*row_index][*col_index] = Cell::Number(i);
                            made_progress = true;
                        }
                    }
                }
            }
            made_progress
        }

        fn insert_forced_constraints_in_squares(board: &mut Board) -> bool {
            let mut made_progress = false;
            for (square_row_index, square_col_index) in MIDDLE_OF_SQUARE_INDEXES {
                let mut occurrences = FxHashMap::default();
                for (offset_y, offset_x) in OFFSETS {
                    let row_index = (square_row_index + offset_y) as usize;
                    let col_index = (square_col_index + offset_x) as usize;

                    let cell = &board.0[row_index][col_index];
                    match cell {
                        Cell::Number(n) => {
                            occurrences
                                .entry(*n)
                                .or_insert(Vec::with_capacity(9))
                                .push((row_index, col_index));
                        }
                        Cell::Constrained(cons) => {
                            for c in cons {
                                occurrences
                                    .entry(*c)
                                    .or_insert(Vec::with_capacity(9))
                                    .push((row_index, col_index));
                            }
                        }
                        _ => {}
                    };
                }
                for i in 1..=9 {
                    if let Some(indexes) = occurrences.get(&i) {
                        if indexes.len() == 1 {
                            let (row_index, col_index) = indexes.last().unwrap();
                            board.0[*row_index][*col_index] = Cell::Number(i);
                            made_progress = true;
                        }
                    }
                }
            }
            made_progress
        }

        let mut made_progress = false;

        made_progress |= {
            let (status, made_progress) = insert_obviously_forced_constraints(&mut self.board);
            if let SolveStatusProgress::NotSolvable = status {
                return SolveStatusProgress::NotSolvable;
            }
            made_progress
        };

        insert_forced_constraints_in_row(&mut self.board);
        insert_forced_constraints_in_col(&mut self.board);
        insert_forced_constraints_in_squares(&mut self.board);

        if made_progress {
            return SolveStatusProgress::MadeProgress;
        }

        SolveStatusProgress::Stalling
    }
}

