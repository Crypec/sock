use rustc_hash::FxHashMap;
use std::assert_matches::*;

use crate::board::{
    Board, BoardIter, BoardWithConstraints, Cell, CellWithConstraints, ColIter, ConstraintList, RowIter, SquareIter,
    SudokuNum,
};

#[derive(Debug, Copy, Clone)]
pub struct BoardNotSolvableError;

pub struct Solver {
    pub board: Board,

    hidden_sets_row_cache: FxHashMap<ConstraintList, Vec<(usize, usize)>>,
    hidden_sets_col_cache: FxHashMap<ConstraintList, Vec<(usize, usize)>>,
    hidden_sets_square_cache: FxHashMap<ConstraintList, Vec<(usize, usize)>>,

    made_progress: bool,

    // most constraining value
    mcv_candidate: (u8, Option<(usize, usize)>),

    col_missing: [ConstraintList; 9],
    row_missing: [ConstraintList; 9],
    square_missing: [ConstraintList; 9],
}

impl Solver {
    // because the are only 9 possible numbers in sudoku
    const MAX_MCV_CANDIDATE_LEN: u8 = 9;

    pub fn new(board: Board) -> Self {
        Self {
            board,

            hidden_sets_row_cache: FxHashMap::default(),
            hidden_sets_col_cache: FxHashMap::default(),
            hidden_sets_square_cache: FxHashMap::default(),

            mcv_candidate: (Self::MAX_MCV_CANDIDATE_LEN, None),

            made_progress: false,
            col_missing: [ConstraintList::full(); 9],
            row_missing: [ConstraintList::full(); 9],
            square_missing: [ConstraintList::full(); 9],
        }
    }

    pub fn solve(&mut self) -> Result<Board, BoardNotSolvableError> {
        self.partially_propagate_constraints();
        self.solve_internal(0)
    }

    fn dbg_missing(&self) {
        dbg!(&self.row_missing);
        dbg!(&self.col_missing);
        dbg!(&self.square_missing);
    }

    fn solve_internal(&mut self, depth: usize) -> Result<Board, BoardNotSolvableError> {
        while !self.is_solved() && self.made_progress {
            self.made_progress = false;
            self.insert_naked_singles()?;
            self.compute_hidden_subsets();
        }
        if self.board.is_solved() {
            return Ok(self.board.clone());
        }
        self.solve_dfs(depth)
    }

    fn is_solved(&self) -> bool {
        if self.row_missing.iter().any(|c| !c.is_empty())
            || self.col_missing.iter().any(|c| !c.is_empty())
            || self.square_missing.iter().any(|c| !c.is_empty())
        {
            return false;
        }
        self.board.is_solved()
    }

    fn solve_dfs(&mut self, depth: usize) -> Result<Board, BoardNotSolvableError> {
        if let Some((row_index, col_index)) = self.mcv_candidate.1 {
            self.invalidate_mcv_candidate();
            let cell = &self.board.0[row_index][col_index];
            if *cell == Cell::Free {
                let cons = self.constraints_at(row_index, col_index);
                let old_row_missing = self.row_missing;
                let old_col_missing = self.col_missing;
                let old_square_misisng = self.square_missing;

                for c in cons {
                    let old_board = self.board.clone();
                    self.insert_and_forward_propagate(c, row_index, col_index);

                    match self.solve_internal(depth + 1) {
                        Ok(board) => return Ok(board),
                        Err(BoardNotSolvableError) => {
                            self.board = old_board;
                            self.row_missing = old_row_missing;
                            self.col_missing = old_col_missing;
                            self.square_missing = old_square_misisng;
                        }
                    }
                }
                return Err(BoardNotSolvableError);
            }
        }
        for (row_index, col_index) in BoardIter::new() {
            let cell = &self.board.0[row_index][col_index];
            if *cell == Cell::Free {
                let cons = self.constraints_at(row_index, col_index);
                let old_row_missing = self.row_missing;
                let old_col_missing = self.col_missing;
                let old_square_misisng = self.square_missing;

                for c in cons {
                    let old_board = self.board.clone();
                    self.insert_and_forward_propagate(c, row_index, col_index);

                    match self.solve_internal(depth + 1) {
                        Ok(board) => return Ok(board),
                        Err(BoardNotSolvableError) => {
                            self.board = old_board;
                            self.row_missing = old_row_missing;
                            self.col_missing = old_col_missing;
                            self.square_missing = old_square_misisng;
                        }
                    }
                }
                return Err(BoardNotSolvableError);
            }
        }
        Err(BoardNotSolvableError)
    }

    fn invalidate_mcv_candidate(&mut self) {
        self.mcv_candidate.0 = Self::MAX_MCV_CANDIDATE_LEN;
        self.mcv_candidate.1 = None;
    }

    const fn constraints_at(&self, row_index: usize, col_index: usize) -> ConstraintList {
        let square_index = Self::calculate_square_index(row_index, col_index);

        let row_missing = self.row_missing[row_index];
        let col_missing = self.col_missing[col_index];
        let square_missing = self.square_missing[square_index];

        ConstraintList::intersection(row_missing, col_missing, square_missing)
    }

    fn insert_naked_singles(&mut self) -> Result<(), BoardNotSolvableError> {
        for (row_index, col_index) in BoardIter::new() {
            let cell = &self.board.0[row_index][col_index];
            if *cell == Cell::Free {
                let cons = self.constraints_at(row_index, col_index);
                if cons.is_empty() {
                    return Err(BoardNotSolvableError);
                } else if let Some(num) = cons.naked_single() {
                    self.insert_and_forward_propagate(num, row_index, col_index);
                }
            }
        }
        Ok(())
    }

    fn clear_row_subset_cache(&mut self) {
        for indexes in &mut self.hidden_sets_row_cache.values_mut() {
            indexes.clear();
        }
    }
    fn clear_col_subset_cache(&mut self) {
        for indexes in &mut self.hidden_sets_col_cache.values_mut() {
            indexes.clear();
        }
    }

    fn clear_square_subset_cache(&mut self) {
        for indexes in &mut self.hidden_sets_square_cache.values_mut() {
            indexes.clear();
        }
    }

    fn compute_hidden_subsets(&mut self) {
        for i in 0..9 {
            // hidden subsets in row
            {
                self.clear_row_subset_cache();
                for (row_index, col_index) in RowIter::new(i) {
                    let cell = &self.board.0[row_index][col_index];
                    if *cell == Cell::Free {
                        let cons = self.constraints_at(row_index, col_index);
                        for k in 1..=4 {
                            let subsets = cons.combinations(k);
                            for s in subsets {
                                let occurrences = self
                                    .hidden_sets_row_cache
                                    .entry(s)
                                    .or_insert_with(|| Vec::with_capacity(9));
                                occurrences.push((row_index, col_index));
                            }
                        }
                    }
                }
                self.insert_constraints_from_hidden_sets_row_cache();
            }

            // hidden singles in columns
            {
                self.clear_col_subset_cache();
                for (row_index, col_index) in ColIter::new(i) {
                    let cell = &self.board.0[row_index][col_index];
                    if *cell == Cell::Free {
                        let cons = self.constraints_at(row_index, col_index);
                        for k in 1..=4 {
                            let subsets = cons.combinations(k);
                            for s in subsets {
                                let occurrences = self
                                    .hidden_sets_col_cache
                                    .entry(s)
                                    .or_insert_with(|| Vec::with_capacity(9));
                                occurrences.push((row_index, col_index));
                            }
                        }
                    }
                }
                self.insert_constraints_from_hidden_sets_col_cache();
            }

            // hidden singles in squares
            {
                self.clear_square_subset_cache();
                for (row_index, col_index) in SquareIter::new(i) {
                    let cell = &self.board.0[row_index][col_index];
                    if *cell == Cell::Free {
                        let cons = self.constraints_at(row_index, col_index);
                        for k in 1..=4 {
                            let subsets = cons.combinations(k);
                            for s in subsets {
                                let occurrences = self
                                    .hidden_sets_square_cache
                                    .entry(s)
                                    .or_insert_with(|| Vec::with_capacity(9));
                                occurrences.push((row_index, col_index));
                            }
                        }
                    }
                }
                self.insert_constraints_from_hidden_sets_square_cache();
            }
        }
    }

    fn insert_constraints_from_hidden_sets_row_cache(&mut self) {
        unsafe {
            let this = self as *mut Self;
            Self::insert_constraints_from_hidden_sets_cache_raw(this, &self.hidden_sets_row_cache);
        }
    }

    fn insert_constraints_from_hidden_sets_col_cache(&mut self) {
        unsafe {
            let this = self as *mut Self;
            Self::insert_constraints_from_hidden_sets_cache_raw(this, &self.hidden_sets_col_cache);
        }
    }
    fn insert_constraints_from_hidden_sets_square_cache(&mut self) {
        unsafe {
            let this = self as *mut Self;
            Self::insert_constraints_from_hidden_sets_cache_raw(this, &self.hidden_sets_square_cache);
        }
    }

    unsafe fn insert_constraints_from_hidden_sets_cache_raw(
        this: *mut Self,
        hidden_sets_cache: &FxHashMap<ConstraintList, Vec<(usize, usize)>>,
    ) {
        for (cons, occurrences) in hidden_sets_cache {
            // we have found a subset of length `k` that is occurring exactly `k` times
            if cons.len() > 1 && cons.len() as usize == occurrences.len() {
                for (row_index, col_index) in occurrences {
                    // also remove from all the squares
                    // we do this here because every cell from the occurrence list could be in different squares
                    // (*this).dbg_missing();
                    // (*this).remove_all_cons_at_pos(*row_index, *col_index, *cons);
                    // (*this).insert_cons_at_pos(*row_index, *col_index, *cons);
                    // println!("we fucked up");
                }
            }
        }

        // Insert hidden singles
        // PERF(Simon): Maybe it is beneficial to first insert all hidden subsets with 2 =< k =< 4.
        // PERF(Simon): This could allow us to find more hidden singles while forward propagation due to a reduced search space.
        for (cons, occurrences) in hidden_sets_cache {
            if let Some(num) = cons.naked_single() && occurrences.len() == 1 {
                let (row_index, col_index) = occurrences[0];
                // dbg!("pretty sure we are screwing up here");
                // dbg!((*this).get_constrained_board());
                // dbg!((num, row_index, col_index));
                (*this).insert_and_forward_propagate(num, row_index, col_index);
                // dbg!("ahhhh I was right");
            }
        }
    }

    const fn calculate_square_index(row_index: usize, col_index: usize) -> usize {
        (row_index / 3) * 3 + col_index / 3
    }

    fn remove_all_cons_at_pos(&mut self, row_index: usize, col_index: usize, to_remove: ConstraintList) {
        let square_index = Self::calculate_square_index(row_index, col_index);

        self.row_missing[row_index].remove_all(to_remove);
        self.col_missing[col_index].remove_all(to_remove);
        self.square_missing[square_index].remove_all(to_remove);
    }

    fn remove_cons_at_pos(&mut self, row_index: usize, col_index: usize, to_remove: SudokuNum) {
        let square_index = Self::calculate_square_index(row_index, col_index);

        self.row_missing[row_index].remove(to_remove);
        self.col_missing[col_index].remove(to_remove);
        self.square_missing[square_index].remove(to_remove);
    }

    fn insert_cons_at_pos(&mut self, row_index: usize, col_index: usize, to_insert: ConstraintList) {}

    // propagate constraints
    pub fn partially_propagate_constraints(&mut self) {
        self.partially_propagate_row_constraints();
        self.partially_propagate_col_constraints();
        self.partially_propagate_square_constraints();
    }

    // PERF(Simon): this is essentially duplicate work if called after `insert_initial_constraints`
    fn partially_propagate_row_constraints(&mut self) {
        for row_index in 0..9 {
            let mut found_nums = ConstraintList::empty();
            for (row_index, col_index) in RowIter::new(row_index) {
                let cell = self.board.0[row_index][col_index];
                if let Cell::Number(n) = cell {
                    found_nums.insert(n);
                }
            }

            self.row_missing[row_index].remove_all(found_nums);
        }
    }
    fn partially_propagate_col_constraints(&mut self) {
        for col_index in 0..9 {
            let mut found_nums = ConstraintList::empty();
            for (row_index, col_index) in ColIter::new(col_index) {
                let cell = &mut self.board.0[row_index][col_index];
                if let Cell::Number(n) = cell {
                    found_nums.insert(*n);
                }
            }
            self.col_missing[col_index].remove_all(found_nums);
        }
    }
    fn partially_propagate_square_constraints(&mut self) {
        for square_index in 0..9 {
            let mut found_nums = ConstraintList::empty();
            for (row_index, col_index) in SquareIter::new(square_index) {
                let cell = &mut self.board.0[row_index][col_index];
                if let Cell::Number(n) = cell {
                    found_nums.insert(*n);
                }
            }
            self.square_missing[square_index].remove_all(found_nums);
        }
    }

    fn insert_and_forward_propagate(&mut self, num: SudokuNum, row_index: usize, col_index: usize) {
        #[cfg(debug_assertions)]
        {
            let cons = self.constraints_at(row_index, col_index);
            debug_assert!(
                cons.contains(num),
                "the placement of a number has to be at least partially valid",
            );
            let cell = self.board.0[row_index][col_index];
            dbg!((row_index, col_index, num));
            debug_assert_matches!(cell, Cell::Free);
        }

        self.board.0[row_index][col_index] = Cell::Number(num);
        self.made_progress = true;

        self.remove_cons_at_pos(row_index, col_index, num);

        // PERF(Simon): Maybe we can fuse these loops to avoid doing duplicate iteration over the same data
        // PERF(Simon): I'm not sure the compiler is smart enough to do this optimization on it's own.
        for (row_index, col_index) in RowIter::new(row_index) {
            let cell = &mut self.board.0[row_index][col_index];
            if *cell == Cell::Free {
                let cons = self.constraints_at(row_index, col_index);
                self.maybe_update_mcv_candidate(row_index, col_index, cons.len() as u8);
                if let Some(num) = cons.naked_single() {
                    self.insert_and_forward_propagate(num, row_index, col_index);
                }
            }
        }

        for (row_index, col_index) in ColIter::new(col_index) {
            let cell = &mut self.board.0[row_index][col_index];
            if *cell == Cell::Free {
                let cons = self.constraints_at(row_index, col_index);
                if let Some(num) = cons.naked_single() {
                    self.insert_and_forward_propagate(num, row_index, col_index);
                }
            }
        }

        let square_index = Self::calculate_square_index(row_index, col_index);
        for (row_index, col_index) in SquareIter::new(square_index) {
            let cell = &mut self.board.0[row_index][col_index];
            if *cell == Cell::Free {
                let cons = self.constraints_at(row_index, col_index);
                if let Some(num) = cons.naked_single() {
                    self.insert_and_forward_propagate(num, row_index, col_index);
                }
            }
        }
    }

    #[inline]
    fn maybe_update_mcv_candidate(&mut self, row_index: usize, col_index: usize, proposed_len: u8) {
        let candidate_len = self.mcv_candidate.0;
        if proposed_len < candidate_len {
            self.mcv_candidate = (proposed_len, Some((row_index, col_index)));
        }
    }

    pub fn is_subset(&self, other: &Board) -> bool {
        let mut is_subset = true;
        for (row_index, col_index) in BoardIter::new() {
            let cell = &self.board.0[row_index][col_index];
            let needle = &other.0[row_index][col_index];
            match (cell, needle) {
                (Cell::Number(n1), Cell::Number(n2)) if n1 == n2 => continue,
                (Cell::Free, Cell::Number(n)) => {
                    let cons = self.constraints_at(row_index, col_index);
                    if cons.contains(*n) {
                        continue;
                    }
                }
                (_, _) => {
                    is_subset = false;
                }
            }
        }
        is_subset
    }

    pub fn get_constrained_board(&self) -> BoardWithConstraints {
        let mut new_board = BoardWithConstraints::new();
        for (row_index, col_index) in BoardIter::new() {
            let old_cell = self.board.0[row_index][col_index];
            let new_cell = match old_cell {
                Cell::Number(n) => CellWithConstraints::Number(n),
                Cell::Free => {
                    let cons = self.constraints_at(row_index, col_index);
                    CellWithConstraints::Constrained(cons)
                }
            };
            new_board.0[row_index][col_index] = new_cell;
        }
        new_board
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::parse_board;
    use crate::*;
    use std::assert_matches::assert_matches;

    #[test]
    fn test_calculate_square_index() {
        let expected: [[usize; 9]; 9] = [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [3, 3, 3, 4, 4, 4, 5, 5, 5],
            [6, 6, 6, 7, 7, 7, 8, 8, 8],
            [6, 6, 6, 7, 7, 7, 8, 8, 8],
            [6, 6, 6, 7, 7, 7, 8, 8, 8],
        ];
        for row_index in 0..9 {
            for col_index in 0..9 {
                let square_index = Solver::calculate_square_index(row_index, col_index);
                assert!(square_index < 9);

                let expected_square_index = expected[row_index][col_index];
                assert_eq!(square_index, expected_square_index);
            }
        }
    }

    #[test]
    fn test_insert_and_forward_propagate() {
        let test_board = parse_board(vec![
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
        let mut solver = Solver::new(test_board);
        solver.partially_propagate_constraints();
        solver.insert_and_forward_propagate(SudokuNum::Three, 8, 0);

        assert_eq!(solver.board.0[8][0], Cell::Number(SudokuNum::Three));
        for (row_index, col_index) in RowIter::new(8) {
            let cons = solver.constraints_at(row_index, col_index);
            assert_eq!(cons.contains(SudokuNum::Three), false);
        }

        for (row_index, col_index) in ColIter::new(0) {
            let cons = solver.constraints_at(row_index, col_index);
            assert_eq!(cons.contains(SudokuNum::Three), false);
        }

        for (row_index, col_index) in SquareIter::new(6) {
            let cons = solver.constraints_at(row_index, col_index);
            assert_eq!(cons.contains(SudokuNum::Three), false);
        }
    }

    #[test]
    fn test_solve_board_1() {
        let test_board = parse_board(vec![
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

        let test_board_solution = parse_board(vec![
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
        assert!(test_board_solution.is_solved());

        let mut solver = Solver::new(test_board);
        let res = solver.solve();
        dbg!(&solver.board);
        dbg!(&solver.is_subset(&test_board_solution));
        dbg!(&solver.get_constrained_board());
        dbg!(&solver.row_missing);
        dbg!(&solver.col_missing);
        dbg!(&solver.square_missing);
        assert_matches!(res, Ok(b) if b.is_solved() && b == test_board_solution)
    }

    #[test]
    fn test_solve_board_4() {
        let board_4 = parse_board(vec![
            vec!['.', '.', '.', '.', '.', '.', '.', '1', '2'],
            vec!['.', '.', '8', '.', '3', '.', '.', '.', '.'],
            vec!['.', '.', '.', '.', '.', '.', '.', '4', '.'],
            vec!['1', '2', '.', '5', '.', '.', '.', '.', '.'],
            vec!['.', '.', '.', '.', '.', '4', '7', '.', '.'],
            vec!['.', '6', '.', '.', '.', '.', '.', '.', '.'],
            vec!['5', '.', '7', '.', '.', '.', '3', '.', '.'],
            vec!['.', '.', '.', '6', '2', '.', '.', '.', '.'],
            vec!['.', '.', '.', '1', '.', '.', '.', '.', '.'],
        ]);
        let board_4_solution = parse_board(vec![
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
        assert!(board_4_solution.is_solved());

        let mut solver = Solver::new(board_4);
        let res = solver.solve();
        assert_matches!(res, Ok(b) if b.is_solved() && b == board_4_solution)
    }

    #[test]
    fn test_solve_hard_leetcode() {
        let hard_leetcode = parse_board(vec![
            vec!['.', '.', '.', '.', '.', '7', '.', '.', '9'],
            vec!['.', '4', '.', '.', '8', '1', '2', '.', '.'],
            vec!['.', '.', '.', '9', '.', '.', '.', '1', '.'],
            vec!['.', '.', '5', '3', '.', '.', '.', '7', '2'],
            vec!['2', '9', '3', '.', '.', '.', '.', '5', '.'],
            vec!['.', '.', '.', '.', '.', '5', '3', '.', '.'],
            vec!['8', '.', '.', '.', '2', '3', '.', '.', '.'],
            vec!['7', '.', '.', '.', '5', '.', '.', '4', '.'],
            vec!['5', '3', '1', '.', '7', '.', '.', '.', '.'],
        ]);

        let hard_leetcode_solution = parse_board(vec![
            vec!['3', '1', '2', '5', '4', '7', '8', '6', '9'],
            vec!['9', '4', '7', '6', '8', '1', '2', '3', '5'],
            vec!['6', '5', '8', '9', '3', '2', '7', '1', '4'],
            vec!['1', '8', '5', '3', '6', '4', '9', '7', '2'],
            vec!['2', '9', '3', '7', '1', '8', '4', '5', '6'],
            vec!['4', '7', '6', '2', '9', '5', '3', '8', '1'],
            vec!['8', '6', '4', '1', '2', '3', '5', '9', '7'],
            vec!['7', '2', '9', '8', '5', '6', '1', '4', '3'],
            vec!['5', '3', '1', '4', '7', '9', '6', '2', '8'],
        ]);
        assert!(hard_leetcode_solution.is_solved());

        let mut solver = Solver::new(hard_leetcode);
        let res = solver.solve();
        assert_matches!(res, Ok(b) if b.is_solved() && b == hard_leetcode_solution)
    }

    fn test_solver() {
        // let mut _codegolf = parse_board(vec![
        //     vec!['.', '.', '.', '7', '.', '.', '.', '.', '.'],
        //     vec!['1', '.', '.', '.', '.', '.', '.', '.', '.'],
        //     vec!['.', '.', '.', '4', '3', '.', '2', '.', '.'],
        //     vec!['.', '.', '.', '.', '.', '.', '.', '.', '6'],
        //     vec!['.', '.', '.', '5', '.', '9', '.', '.', '.'],
        //     vec!['.', '.', '.', '.', '.', '.', '4', '1', '8'],
        //     vec!['.', '.', '.', '.', '8', '1', '.', '.', '.'],
        //     vec!['.', '.', '2', '.', '.', '.', '.', '5', '.'],
        //     vec!['.', '4', '.', '.', '.', '.', '3', '.', '.'],
        // ]);
    }
}
