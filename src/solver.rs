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

    _col_missing: [ConstraintList; 9],
    _row_missing: [ConstraintList; 9],
    _square_missing: [ConstraintList; 9],
}

impl Solver {
    pub fn new(board: Board) -> Self {
        Self {
            board,

            hidden_sets_row_cache: FxHashMap::default(),
            hidden_sets_col_cache: FxHashMap::default(),
            hidden_sets_square_cache: FxHashMap::default(),

            made_progress: false,

            _col_missing: std::array::from_fn(|_| ConstraintList::full()),
            _row_missing: std::array::from_fn(|_| ConstraintList::full()),
            _square_missing: std::array::from_fn(|_| ConstraintList::full()),
        }
    }

    pub fn solve(&mut self) -> Result<Board, BoardNotSolvableError> {
        self.insert_initial_constraints();
        self.partially_propagate_constraints();
        self.solve_internal()?;
        Ok(self.board.clone())
    }

    fn solve_internal(&mut self) -> Result<(), BoardNotSolvableError> {
        while !self.board.is_solved() {
            // dbg!(self.made_progress);
            // dbg!(&self.board);
            while self.made_progress {
                self.made_progress = false;
                self.insert_naked_singles()?;
                // dbg!(&self.board);
                // self.insert_hidden_singles();
                // self.insert_hidden_subsets();
            }
            self.solve_dfs()?;
        }
        Ok(())
    }

    fn solve_dfs(&mut self) -> Result<(), BoardNotSolvableError> {
        // panic!("failed to solve in dfs {:?}", self.board);
        for (row_index, col_index) in BoardIter::new() {
            let cell = self.board.0[row_index][col_index];
            if let Cell::Constrained(cons) = cell {
                for c in &cons {
                    let old_board = self.board.clone();
                    self.insert_and_forward_propagate(c, row_index, col_index);
                    if self.solve_internal().is_ok() {
                        return Ok(());
                    }
                    self.board = old_board;
                }
                return Err(BoardNotSolvableError);
            }
        }
        // panic!("board not solvable in `solve_board_dfs` :: end of function");
        Err(BoardNotSolvableError)
    }

    fn insert_naked_singles(&mut self) -> Result<(), BoardNotSolvableError> {
        for (row_index, col_index) in BoardIter::new() {
            let cell = &self.board.0[row_index][col_index];
            match cell {
                Cell::Constrained(cons) if cons.is_empty() => {
                    // panic!("board not solvable in `insert_naked_singles`");
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

    fn clear_subset_caches(&mut self) {
        self.hidden_sets_row_cache.clear();
        self.hidden_sets_col_cache.clear();
        self.hidden_sets_square_cache.clear();
        self.clear_row_subset_cache();
        self.clear_col_subset_cache();
        self.clear_square_subset_cache();
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

    fn insert_hidden_singles(&mut self) {
        self.clear_subset_caches();
        for i in 0..9 {
            self.clear_row_subset_cache();
            // dbg!(&self.hidden_sets_row_cache);
            for (row_index, col_index) in RowIter::new(i) {
                let cell = &self.board.0[row_index][col_index];
                if let Cell::Constrained(cons) = cell {
                    for k in 1..=4 {
                        let subsets = cons.combinations(k);
                        for c in subsets {
                            self.hidden_sets_row_cache
                                .entry(c)
                                .and_modify(|p| p.push((row_index, col_index)))
                                .or_insert_with(|| Vec::with_capacity(9));
                        }
                    }
                }
            }
            self.insert_hidden_subsets(&self.hidden_sets_row_cache.clone());

            for (row_index, col_index) in ColIter::new(i) {
                let cell = &self.board.0[row_index][col_index];
                if let Cell::Constrained(cons) = cell {
                    for k in 1..=4 {
                        let subsets = cons.combinations(k);
                        for c in subsets {
                            self.hidden_sets_col_cache
                                .entry(c)
                                .and_modify(|p| p.push((row_index, col_index)))
                                .or_insert_with(|| Vec::with_capacity(9));
                        }
                    }
                }
            }
            self.insert_hidden_subsets(&self.hidden_sets_col_cache.clone());
            for (row_index, col_index) in SquareIter::new(i) {
                let cell = &self.board.0[row_index][col_index];
                if let Cell::Constrained(cons) = cell {
                    for k in 1..=4 {
                        let subsets = cons.combinations(k);
                        for c in subsets {
                            self.hidden_sets_square_cache
                                .entry(c)
                                .and_modify(|p| p.push((row_index, col_index)))
                                .or_insert_with(|| Vec::with_capacity(9));
                        }
                    }
                }
            }
            self.insert_hidden_subsets(&self.hidden_sets_square_cache.clone());
        }
        // dbg!(&self.hidden_sets_row_cache);
        // dbg!(&self.hidden_sets_col_cache);
        // dbg!(&self.hidden_sets_square_cache);
    }

    pub fn insert_hidden_subsets(&mut self, subsets: &FxHashMap<ConstraintList, Vec<(usize, usize)>>) {
        let max_capacity = subsets.len();
        let mut to_insert = Vec::with_capacity(max_capacity);
        let mut to_override = Vec::with_capacity(max_capacity);

        for (cons, indexes) in subsets {
            if cons.len() == 1 && indexes.len() == 1 {
                let (row_index, col_index) = indexes[0];
                let num = cons.first().unwrap();
                to_insert.push((num, (row_index, col_index)));
            }
            for i in 2..=4usize {
                if cons.len() as usize == i && indexes.len() == i {
                    for (row_index, col_index) in indexes {
                        to_override.push((*cons, (*row_index, *col_index)));
                    }
                }
            }
        }

        if !to_insert.is_empty() {
            // dbg!(&self.board);
            // dbg!(&to_insert);
            // panic!("stop");
            // std::thread::sleep(std::time::Duration::from_secs(1));
        }
        if !to_override.is_empty() {
            // dbg!(&self.board);
            // dbg!(&to_override);
            // panic!("stop");
            // std::thread::sleep(std::time::Duration::from_secs(1));
        }

        // for (cons, (row_index, col_index)) in to_override {
        //     self.board.0[row_index][col_index] = Cell::Constrained(cons);
        // }

        for (num, (row_index, col_index)) in to_insert {
            self.insert_and_forward_propagate(num, row_index, col_index);
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

        self.partially_propagate_constraints();
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
