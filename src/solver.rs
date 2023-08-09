use rustc_hash::FxHashMap;

use crate::board::{Board, BoardIter, Cell, ColIter, ConstraintList, RowIter, SquareIter, SudokuNum};

#[derive(Debug, Copy, Clone)]
pub struct BoardNotSolvableError;

pub struct Solver {
    pub board: Board,

    hidden_sets_row_cache: FxHashMap<ConstraintList, Vec<(usize, usize)>>,
    hidden_sets_col_cache: FxHashMap<ConstraintList, Vec<(usize, usize)>>,
    hidden_sets_square_cache: FxHashMap<ConstraintList, Vec<(usize, usize)>>,

    made_progress: bool,
    // _col_missing: [ConstraintList; 9],
    // _row_missing: [ConstraintList; 9],
    // _square_missing: [ConstraintList; 9],
}

impl Solver {
    pub fn new(board: Board) -> Self {
        Self {
            board,

            hidden_sets_row_cache: FxHashMap::default(),
            hidden_sets_col_cache: FxHashMap::default(),
            hidden_sets_square_cache: FxHashMap::default(),

            made_progress: false,
            // _col_missing: std::array::from_fn(|_| ConstraintList::full()),
            // _row_missing: std::array::from_fn(|_| ConstraintList::full()),
            // _square_missing: std::array::from_fn(|_| ConstraintList::full()),
        }
    }

    pub fn solve(&mut self) -> Result<Board, BoardNotSolvableError> {
        self.insert_initial_constraints();
        self.partially_propagate_constraints();
        self.solve_internal(0)
    }

    fn solve_internal(&mut self, depth: usize) -> Result<Board, BoardNotSolvableError> {
        while !self.board.is_solved() && self.made_progress {
            self.made_progress = false;
            self.insert_naked_singles()?;
            self.compute_hidden_subsets();
        }
        if self.board.is_solved() {
            return Ok(self.board.clone());
        }
        self.solve_dfs(depth)
    }

    fn solve_dfs(&mut self, depth: usize) -> Result<Board, BoardNotSolvableError> {
        for (row_index, col_index) in BoardIter::new() {
            let cell = &self.board.0[row_index][col_index];
            if let Cell::Constrained(cons) = cell {
                for c in cons {
                    let old_board = self.board.clone();
                    self.insert_and_forward_propagate(c, row_index, col_index);

                    match self.solve_internal(depth + 1) {
                        Ok(board) => return Ok(board),
                        Err(BoardNotSolvableError) => {
                            self.board = old_board;
                        }
                    }
                }
                return Err(BoardNotSolvableError);
            }
        }
        Err(BoardNotSolvableError)
    }

    fn insert_naked_singles(&mut self) -> Result<(), BoardNotSolvableError> {
        for (row_index, col_index) in BoardIter::new() {
            let cell = &self.board.0[row_index][col_index];
            match cell {
                Cell::Constrained(cons) if cons.is_empty() => {
                    return Err(BoardNotSolvableError);
                }
                Cell::Constrained(cons) if cons.is_naked_single() => {
                    let num = cons.first().unwrap();
                    self.insert_and_forward_propagate(num, row_index, col_index);
                }
                _ => continue,
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
                    if let Cell::Constrained(cons) = cell {
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
                self.insert_from_hidden_singles_row_cache();
            }

            // hidden singles in columns
            {
                self.clear_col_subset_cache();
                for (row_index, col_index) in ColIter::new(i) {
                    let cell = &self.board.0[row_index][col_index];
                    if let Cell::Constrained(cons) = cell {
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
                self.insert_from_hidden_singles_col_cache();
            }

            // hidden singles in squares
            {
                self.clear_square_subset_cache();
                for (row_index, col_index) in SquareIter::new(i) {
                    let cell = &self.board.0[row_index][col_index];
                    if let Cell::Constrained(cons) = cell {
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
                self.insert_from_hidden_singles_square_cache();
            }
        }
    }

    fn insert_from_hidden_singles_row_cache(&mut self) {
        unsafe {
            let this = self as *mut Self;
            for (cons, occurrences) in &self.hidden_sets_row_cache {
                if cons.len() > 1 && cons.len() as usize == occurrences.len() {
                    // NOTE(Simon): check that we really are in the same row
                    #[cfg(debug_assertions)]
                    for w in occurrences.windows(2) {
                        let (r0, _) = w[0];
                        let (r1, _) = w[1];

                        debug_assert_eq!(r0, r1);
                    }

                    let (row_index, col_index) = occurrences[0];
                    for i in 0..9 {
                        // remove the constraint for all the cells in the same row
                        if let Cell::Constrained(mut old) = self.board.0[row_index][i] {
                            old.remove_all(*cons);
                        }

                        // remove the constraint for all the cells in the same column
                        if let Cell::Constrained(mut old) = self.board.0[i][col_index] {
                            old.remove_all(*cons);
                        }
                    }

                    for (row_index, col_index) in occurrences {
                        // also remove from all the squares
                        // we do this here because every cell from the occurrence list could be in different squares
                        {
                            let square_index = Self::calculate_square_index(*row_index, *col_index);
                            for (row_index, col_index) in SquareIter::new(square_index) {
                                if let Cell::Constrained(mut old) = self.board.0[row_index][col_index] {
                                    old.remove_all(*cons);
                                }
                            }
                        }

                        // NOTE(Simon): the variable never read warning here makes me a bit suspicious
                        #[allow(unused_variables)]
                        #[allow(unused_assignments)]
                        if let Cell::Constrained(mut old) = &mut (*this).board.0[*row_index][*col_index] {
                            old = *cons;
                        }
                    }
                }
            }

            // Insert hidden singles
            // PERF(Simon): Maybe it is beneficial to first insert all hidden subsets with 2 =< k =< 4.
            // PERF(Simon): This could allow us to find more hidden singles while forward propagation due to a reduced search space.
            for (cons, occurrences) in &self.hidden_sets_row_cache {
                if cons.is_naked_single() && occurrences.len() == 1 {
                    let (row_index, col_index) = occurrences[0];
                    let number = cons.first().unwrap();
                    (*this).insert_and_forward_propagate(number, row_index, col_index);
                }
            }
        }
    }

    fn insert_from_hidden_singles_col_cache(&mut self) {
        unsafe {
            let this = self as *mut Self;
            for (cons, occurrences) in &self.hidden_sets_col_cache {
                if cons.len() > 1 && cons.len() as usize == occurrences.len() {
                    // NOTE(Simon): check that we really are in the same row
                    #[cfg(debug_assertions)]
                    for w in occurrences.windows(2) {
                        let (_, c0) = w[0];
                        let (_, c1) = w[1];

                        debug_assert_eq!(c0, c1);
                    }

                    let (row_index, col_index) = occurrences[0];
                    for i in 0..9 {
                        // remove the constraint for all the cells in the same row
                        if let Cell::Constrained(mut old) = self.board.0[row_index][i] {
                            old.remove_all(*cons);
                        }

                        // remove the constraint for all the cells in the same column
                        if let Cell::Constrained(mut old) = self.board.0[i][col_index] {
                            old.remove_all(*cons);
                        }
                    }

                    for (row_index, col_index) in occurrences {
                        // also remove from all the squares
                        // we do this here because every cell from the occurrence list could be in different squares
                        {
                            let square_index = Self::calculate_square_index(*row_index, *col_index);
                            for (row_index, col_index) in SquareIter::new(square_index) {
                                if let Cell::Constrained(mut old) = self.board.0[row_index][col_index] {
                                    old.remove_all(*cons);
                                }
                            }
                        }

                        // NOTE(Simon): the variable never read warning here makes me a bit suspicious
                        #[allow(unused_variables)]
                        #[allow(unused_assignments)]
                        if let Cell::Constrained(mut old) = &mut (*this).board.0[*row_index][*col_index] {
                            old = *cons;
                        }
                    }
                }
            }

            // Insert hidden singles
            // PERF(Simon): Maybe it is beneficial to first insert all hidden subsets with 2 =< k =< 4.
            // PERF(Simon): This could allow us to find more hidden singles while forward propagation due to a reduced search space.
            for (cons, occurrences) in &self.hidden_sets_col_cache {
                if cons.is_naked_single() && occurrences.len() == 1 {
                    let (row_index, col_index) = occurrences[0];
                    let number = cons.first().unwrap();
                    (*this).insert_and_forward_propagate(number, row_index, col_index);
                }
            }
        }
    }

    fn insert_from_hidden_singles_square_cache(&mut self) {
        unsafe {
            let this = self as *mut Self;
            for (cons, occurrences) in &self.hidden_sets_square_cache {
                if cons.len() > 1 && cons.len() as usize == occurrences.len() {
                    let (row_index, col_index) = occurrences[0];
                    for i in 0..9 {
                        // remove the constraint for all the cells in the same row
                        if let Cell::Constrained(mut old) = self.board.0[row_index][i] {
                            old.remove_all(*cons);
                        }

                        // remove the constraint for all the cells in the same column
                        if let Cell::Constrained(mut old) = self.board.0[i][col_index] {
                            old.remove_all(*cons);
                        }
                    }

                    for (row_index, col_index) in occurrences {
                        // also remove from all the squares
                        // we do this here because every cell from the occurrence list could be in different squares
                        {
                            let square_index = Self::calculate_square_index(*row_index, *col_index);
                            for (row_index, col_index) in SquareIter::new(square_index) {
                                if let Cell::Constrained(mut old) = self.board.0[row_index][col_index] {
                                    old.remove_all(*cons);
                                }
                            }
                        }

                        // NOTE(Simon): the variable never read warning here makes me a bit suspicious
                        #[allow(unused_variables)]
                        #[allow(unused_assignments)]
                        if let Cell::Constrained(mut old) = &mut (*this).board.0[*row_index][*col_index] {
                            old = *cons;
                        }
                    }
                }
            }

            // Insert hidden singles
            // PERF(Simon): Maybe it is beneficial to first insert all hidden subsets with 2 =< k =< 4.
            // PERF(Simon): This could allow us to find more hidden singles while forward propagation due to a reduced search space.
            for (cons, occurrences) in &self.hidden_sets_square_cache {
                if cons.is_naked_single() && occurrences.len() == 1 {
                    let (row_index, col_index) = occurrences[0];
                    let number = cons.first().unwrap();
                    (*this).insert_and_forward_propagate(number, row_index, col_index);
                }
            }
        }
    }

    const fn calculate_square_index(row_index: usize, col_index: usize) -> usize {
        (row_index / 3) * 3 + col_index / 3
    }

    pub fn insert_initial_constraints(&mut self) {
        for row_index in 0..9 {
            let mut possible_nums = ConstraintList::full();
            for col_index in 0..9 {
                let cell = &self.board.0[row_index][col_index];
                if let Cell::Number(n) = &cell {
                    possible_nums.remove(*n);
                }
            }
            for col_index in 0..9 {
                let cell = &mut self.board.0[row_index][col_index];
                if Cell::Free == *cell {
                    *cell = Cell::Constrained(possible_nums);
                }
            }
        }
    }

    // propagate constraints
    pub fn partially_propagate_constraints(&mut self) {
        self.partially_propagate_row_constraints();
        self.partially_propagate_col_constraints();
        self.partially_propagate_square_constraints();
    }

    fn partially_propagate_row_constraints(&mut self) {
        for row_index in 0..9 {
            let mut found_nums = ConstraintList::empty();
            for (row_index, col_index) in RowIter::new(row_index) {
                let cell = self.board.0[row_index][col_index];
                if let Cell::Number(n) = cell {
                    found_nums.insert(n);
                }
            }

            for (row_index, col_index) in RowIter::new(row_index) {
                let cell = &mut self.board.0[row_index][col_index];
                if let Cell::Constrained(cons) = cell {
                    cons.remove_all(found_nums);
                }
            }
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
            for (row_index, col_index) in ColIter::new(col_index) {
                let cell = &mut self.board.0[row_index][col_index];
                if let Cell::Constrained(cons) = cell {
                    cons.remove_all(found_nums);
                }
            }
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
            for (row_index, col_index) in SquareIter::new(square_index) {
                let cell = &mut self.board.0[row_index][col_index];
                if let Cell::Constrained(cons) = cell {
                    cons.remove_all(found_nums);
                }
            }
        }
    }

    fn insert_and_forward_propagate(&mut self, num: SudokuNum, row_index: usize, col_index: usize) {
        self.board.0[row_index][col_index] = Cell::Number(num);
        self.made_progress = true;

        for (row_index, col_index) in RowIter::new(row_index) {
            let cell = &mut self.board.0[row_index][col_index];
            if let Cell::Constrained(cons) = cell {
                cons.remove(num);
                if cons.is_naked_single() {
                    let n = cons.first().unwrap();
                    self.insert_and_forward_propagate(n, row_index, col_index);
                }
            }
        }

        for (row_index, col_index) in ColIter::new(col_index) {
            let cell = &mut self.board.0[row_index][col_index];
            if let Cell::Constrained(cons) = cell {
                cons.remove(num);
                if cons.is_naked_single() {
                    let n = cons.first().unwrap();
                    self.insert_and_forward_propagate(n, row_index, col_index);
                }
            }
        }
        let square_index = Self::calculate_square_index(row_index, col_index);
        for (row_index, col_index) in SquareIter::new(square_index) {
            let cell = &mut self.board.0[row_index][col_index];
            if let Cell::Constrained(cons) = cell {
                cons.remove(num);
                if cons.is_naked_single() {
                    let n = cons.first().unwrap();
                    self.insert_and_forward_propagate(n, row_index, col_index);
                }
            }
        }
    }

    pub fn is_subset(&self, other: &Board) -> bool {
        let mut is_subset = true;
        for (row_index, col_index) in BoardIter::new() {
            let cell = &self.board.0[row_index][col_index];
            let needle = &other.0[row_index][col_index];
            match (cell, needle) {
                (Cell::Number(n1), Cell::Number(n2)) if n1 == n2 => continue,
                (Cell::Constrained(cons), Cell::Number(n)) if cons.contains(*n) => continue,
                (Cell::Constrained(c1), Cell::Constrained(c2)) if c1.contains_all(*c2) => continue,
                (_, _) => {
                    is_subset = false;
                }
            }
        }
        is_subset
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
        solver.insert_initial_constraints();
        solver.partially_propagate_constraints();
        solver.insert_and_forward_propagate(SudokuNum::Three, 8, 0);

        assert_eq!(solver.board.0[8][0], Cell::Number(SudokuNum::Three));
        for (row_index, col_index) in RowIter::new(8) {
            let cell = &solver.board.0[row_index][col_index];
            if let Cell::Constrained(cons) = cell {
                assert_eq!(cons.contains(SudokuNum::Three), false);
            }
        }

        for (row_index, col_index) in ColIter::new(0) {
            let cell = &solver.board.0[row_index][col_index];
            if let Cell::Constrained(cons) = cell {
                assert_eq!(cons.contains(SudokuNum::Three), false);
            }
        }

        for (row_index, col_index) in SquareIter::new(6) {
            let cell = &solver.board.0[row_index][col_index];
            if let Cell::Constrained(cons) = cell {
                assert_eq!(cons.contains(SudokuNum::Three), false);
            }
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
