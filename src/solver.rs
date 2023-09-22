#![allow(clippy::upper_case_acronyms)]

use strum::EnumCount;

use crate::board::{BigBoardPosition, Board, BoardIter, BoxIter, Cell, ColIter, PencilMarks, RowIter, SudokuNum};

use crate::subset_cache::SubSetCache;

use tracing::Origin;

type BoardPosition = BigBoardPosition;

pub const MAX_NUMBER_COUNT: usize = SudokuNum::COUNT;

#[derive(Debug, Copy, Clone)]
pub struct BoardNotSolvableError;

type HiddenSinglesEntryList = [HiddenSingleEntry; MAX_NUMBER_COUNT];
type HiddenPairsCache = SubSetCache<2, 2, MAX_NUMBER_COUNT, Vec<BoardPosition>>;
type Histogram = [u8; MAX_NUMBER_COUNT];

pub struct Solver {
    pub board: Board,

    hidden_pairs_cache: Box<HiddenPairsCache>,

    constraint_stack: Vec<Vec<Constraint>>,

    tier: Tier,

    // most constraining value
    mcv_candidate: (u8, Option<BoardPosition>),

    #[cfg(feature = "tracing")]
    pub trace: tracing::Trace,
}

impl Solver {
    // because the are only 9 possible numbers in sudoku
    const MCV_CANDIDATE_MAX_LEN: u8 = MAX_NUMBER_COUNT as u8;

    #[must_use]
    pub fn new(board: Board) -> Self {
        let cell_count = board.cell_count();
        Self {
            board,

            hidden_pairs_cache: Box::new(SubSetCache::from_fn(|_| Vec::with_capacity(MAX_NUMBER_COUNT))),

            // NOTE(Simon): Not sure if this is correct but at the moment we preallocate space for one constraint per cell
            constraint_stack: vec![Vec::with_capacity(cell_count)],

            mcv_candidate: (Self::MCV_CANDIDATE_MAX_LEN, None),

            tier: Tier::Tier1 { made_progress: false },

            #[cfg(feature = "tracing")]
            trace: tracing::Trace {
                root: board,
                events: vec![],
            },
        }
    }

    pub fn solve(&mut self) -> Result<Board, BoardNotSolvableError> {
        self.update_initial_constraints();
        self.partially_propagate_constraints();

        #[cfg(feature = "tracing")]
        self.trace
            .events
            .push(tracing::Event::PartiallyPropagate { board: self.board });

        self.solve_internal(0)
    }

    fn solve_internal(&mut self, depth: usize) -> Result<Board, BoardNotSolvableError> {
        loop {
            match (self.tier, self.is_solved()) {
                (_, true) => {
                    return Ok(self.board);
                }
                (Tier::Tier1 { made_progress: true }, false) => {
                    self.tier = Tier::Tier1 { made_progress: false };

                    self.insert_naked_singles()?;
                    self.insert_hidden_singles()?;
                }
                (Tier::Tier1 { made_progress: false }, false) => {
                    self.insert_hidden_pairs();
                }
                (
                    Tier::Tier2 {
                        made_progress: true,
                        ran_last_time: true,
                    },
                    false,
                ) => {
                    if self.solve_tier_1()? {
                        return Ok(self.board);
                    }
                    break;
                }
                (
                    Tier::Tier2 {
                        made_progress: false,
                        ran_last_time: true,
                    },
                    _,
                ) => break,
            };
        }
        return self.solve_dfs(depth);
    }

    fn solve_tier_1(&mut self) -> Result<bool, BoardNotSolvableError> {
        let mut is_solved = self.is_solved();
        while let Tier::Tier1 { made_progress: true } = self.tier && !is_solved {
            self.tier = Tier::Tier1 { made_progress: false };

            self.insert_naked_singles()?;
            self.insert_hidden_singles()?;

            is_solved = self.is_solved();
        }
        Ok(is_solved)
    }

    fn is_solved(&self) -> bool {
        self.board.is_solved()
    }

    // If we have exhausted all our options using constraint propagation, run a depth first search
    fn solve_dfs(&mut self, depth: usize) -> Result<Board, BoardNotSolvableError> {
        if let Some(constraints) = self.constraint_stack.pop() {
            // PERF(Simon): we could try to unify these constraints beforehand to check if they are consistent.
            for cons in &constraints {
                match cons {
                    Constraint::NakedPair { marks, positions } => {
                        let combinations = [
                            [(marks.0, positions.0), (marks.1, positions.1)],
                            [(marks.0, positions.1), (marks.0, positions.1)],
                        ];

                        for combi in combinations {
                            match self.try_insert_and_solve(combi, Origin::DFSConstraints, depth) {
                                Ok(board) => return Ok(board),
                                Err(_) => continue,
                            }
                        }
                    }
                }
            }

            // if constraints were not empty and we tried all of them without finding a solution we know the current board is not solvable
            if !constraints.is_empty() {
                return Err(BoardNotSolvableError);
            }
        }

        if let Some(position) = self.mcv_candidate.1 {
            self.invalidate_mcv_candidate();

            let cell = self.board[position];
            if let Cell::Marked(marks) = cell {
                for pm in marks {
                    match self.try_insert_and_solve([(pm, position)], Origin::DFSMCV, depth) {
                        Ok(board) => return Ok(board),
                        Err(_) => continue,
                    };
                }
                return Err(BoardNotSolvableError);
            }
        }

        // if we don't have a candidate for the mcv heuristic we do a linear search for the next free cell
        for position in BoardIter::new() {
            let cell = self.board[position];
            match cell {
                Cell::Marked(marks) => {
                    for pm in marks {
                        match self.try_insert_and_solve([(pm, position)], Origin::DFSTryAll, depth) {
                            Ok(board) => return Ok(board),
                            Err(_) => continue,
                        };
                    }
                    return Err(BoardNotSolvableError);
                }
                Cell::Number(_) => continue,
            };
        }
        Err(BoardNotSolvableError)
    }

    #[inline(always)]
    fn try_insert_and_solve<I>(
        &mut self,
        to_insert_iter: I,
        origin: Origin,
        depth: usize,
    ) -> Result<Board, BoardNotSolvableError>
    where
        I: IntoIterator<Item = (SudokuNum, BoardPosition)>,
    {
        let cell_count = self.board.cell_count();
        self.constraint_stack.push(Vec::with_capacity(cell_count));

        let old_board = self.board;

        for (number, position) in to_insert_iter {
            if self.insert_and_forward_propagate(number, position, origin).is_err() {
                self.restore_state(&old_board);
                return Err(BoardNotSolvableError);
            }
        }

        match self.solve_internal(depth + 1) {
            Ok(board) => {
                #[cfg(feature = "tracing")]
                self.trace.events.push(tracing::Event::Solved { board: self.board });

                Ok(board)
            }
            Err(BoardNotSolvableError) => {
                self.restore_state(&old_board);
                Err(BoardNotSolvableError)
            }
        }
    }

    #[inline(always)]
    fn restore_state(&mut self, old_board: &Board) {
        #[cfg(feature = "tracing")]
        self.trace.events.push(tracing::Event::Restore);

        self.board = *old_board;
        self.constraint_stack.pop();
    }

    fn insert_hidden_singles(&mut self) -> Result<(), BoardNotSolvableError> {
        for i in 0..9 {
            self.insert_hidden_singles_iter(RowIter::new(i))?;
            self.insert_hidden_singles_iter(ColIter::new(i))?;
            self.insert_hidden_singles_iter(BoxIter::new(i))?;
        }
        Ok(())
    }

    fn insert_hidden_singles_iter<I>(&mut self, iter: I) -> Result<(), BoardNotSolvableError>
    where
        I: Iterator<Item = BoardPosition>,
    {
        let mut hidden_singles_cache = [HiddenSingleEntry::None; MAX_NUMBER_COUNT];

        for position in iter {
            let cell = self.board[position];
            if let Cell::Marked(marks) = cell {
                for sudoku_num in marks {
                    Self::maybe_update_hidden_singles_entry(&mut hidden_singles_cache, sudoku_num, position);
                }
            }
        }
        self.insert_hidden_singles_from_entrylist(&hidden_singles_cache)?;

        Ok(())
    }

    fn insert_hidden_singles_from_entrylist(
        &mut self,
        hidden_singles_cache: &HiddenSinglesEntryList,
    ) -> Result<(), BoardNotSolvableError> {
        for (num_index, entry) in hidden_singles_cache.iter().enumerate() {
            if let HiddenSingleEntry::One(position) = entry {
                let num: SudokuNum = (num_index + 1).try_into().expect("failed to convert to sudoku number");
                self.insert_and_forward_propagate(num, *position, Origin::HiddenSingle)?;
            }
        }
        Ok(())
    }

    fn maybe_update_hidden_singles_entry(
        hidden_singles_entry_list: &mut HiddenSinglesEntryList,
        num: SudokuNum,
        position: BoardPosition,
    ) {
        let num_index = (num as usize) - 1;

        let old_entry = unsafe {
            // for some reason llvm is not smart enough to remove this bounds check
            hidden_singles_entry_list.get_unchecked(num_index)
        };
        let new_entry = match old_entry {
            HiddenSingleEntry::None => HiddenSingleEntry::One(position),
            HiddenSingleEntry::One(..) | HiddenSingleEntry::Many => HiddenSingleEntry::Many,
        };

        unsafe {
            *hidden_singles_entry_list.get_unchecked_mut(num_index) = new_entry;
        }
    }

    fn invalidate_mcv_candidate(&mut self) {
        self.mcv_candidate.0 = Self::MCV_CANDIDATE_MAX_LEN;
        self.mcv_candidate.1 = None;
    }

    fn insert_naked_singles(&mut self) -> Result<(), BoardNotSolvableError> {
        for position in BoardIter::new() {
            let cell = self.board[position];
            if let Cell::Marked(marks) = cell {
                if marks.is_empty() {
                    return Err(BoardNotSolvableError);
                } else if let Some(num) = marks.naked_single() {
                    self.insert_and_forward_propagate(num, position, Origin::NakedSingle)?;
                }
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn calculate_box_index(position: BoardPosition) -> usize {
        const BOX_POSITION_LUT: [[usize; 9]; 9] = [
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

        let row_index = position.row_index;
        let col_index = position.col_index;
        unsafe { *BOX_POSITION_LUT.get_unchecked(row_index).get_unchecked(col_index) }
    }

    #[inline(always)]
    fn remove_cons_at_pos(&mut self, to_remove: SudokuNum, position: BoardPosition) {
        let row_index = position.row_index;
        let col_index = position.col_index;
        let box_index = Self::calculate_box_index(position);

        self.remove_cons_at_pos_iter(to_remove, RowIter::new(row_index));
        self.remove_cons_at_pos_iter(to_remove, ColIter::new(col_index));
        self.remove_cons_at_pos_iter(to_remove, BoxIter::new(box_index));
    }

    // PERF(Simon): This function takes up a lot of time in the benchmarks (currently ~ 17% of the total run time)
    // PERF(Simon): I think this could relatively easy be replaced with an unrolled SIMD version
    // PERF(Simon): For this we only wood need to check if the whole row does not contain a `Cell::Number` and if not
    // PERF(Simon): use SIMD to create a bitmask for to remove the number from all cells.
    #[inline(always)]
    fn remove_cons_at_pos_iter<I>(&mut self, to_remove: SudokuNum, iter: I)
    where
        I: Iterator<Item = BoardPosition>,
    {
        for position in iter {
            let cell = &mut self.board[position];
            if let Cell::Marked(marks) = cell {
                marks.remove(to_remove);
            }
        }
    }

    // const fn calculate_box_position(position: BoardPosition) -> usize {
    //     let row_index = position.row_index;
    //     let col_index = position.col_index;

    //     (row_index / 3) * 3 + col_index / 3
    // }

    // PERFORMANCE(Simon): This is kinda redundant work. Because we fully the layout of the boards we could just fill empty cells with full constraints while parsing!
    fn update_initial_constraints(&mut self) {
        for position in BoardIter::new() {
            let cell = &mut self.board[position];
            if let Cell::Marked(_) = cell {
                *cell = Cell::Marked(PencilMarks::full());
            }
        }
    }

    fn partially_propagate_constraints(&mut self) {
        for position in BoardIter::new() {
            let cell = self.board[position];
            if let Cell::Number(number) = cell {
                self.remove_cons_at_pos(number, position);
            }
        }
    }

    fn insert_and_forward_propagate(
        &mut self,
        number: SudokuNum,
        position: BoardPosition,
        _origin: Origin,
    ) -> Result<(), BoardNotSolvableError> {
        #[cfg(feature = "paranoid")]
        {
            use std::assert_matches::*;
            let cell = self.board[position];

            let constrained_board = self.get_constrained_board();

            debug_assert_matches!(cell, Cell::Constrained(marks) if marks.contains(number),
                "the placement of a number has to be at least partially valid :: tried to insert {} into :: {:?} => {:?}{:?}\n{:?}",
            number, (position.row_index, position.col_index), _origin, cons, constrained_board);
        }

        #[cfg(feature = "tracing")]
        {
            let board = self.board;

            self.trace.events.push(tracing::Event::Insert {
                origin: _origin,
                position,
                number,
                board,
            });
        }

        // actually insert the number into the board
        self.board[position] = Cell::Number(number);
        self.tier = Tier::Tier1 { made_progress: true };

        self.remove_cons_at_pos(number, position);

        // PERF(Simon): Maybe we can fuse these loops to avoid doing duplicate iteration over the same data
        // PERF(Simon): I'm not sure the compiler is smart enough to do this optimization on it's own.
        for position in RowIter::new(position.row_index) {
            let cell = self.board[position];
            if let Cell::Marked(marks) = cell {
                if marks.is_empty() {
                    return Err(BoardNotSolvableError);
                }

                // check if we maybe found a more constraining candidate
                self.maybe_update_mcv_candidate(marks.len() as u8, position);

                if let Some(num) = marks.naked_single() {
                    self.insert_and_forward_propagate(num, position, Origin::ForwardPropagate)?;
                }
            }
        }

        for position in ColIter::new(position.col_index) {
            let cell = self.board[position];
            if let Cell::Marked(marks) = cell {
                if marks.is_empty() {
                    return Err(BoardNotSolvableError);
                }

                if let Some(num) = marks.naked_single() {
                    self.insert_and_forward_propagate(num, position, Origin::ForwardPropagate)?;
                }
            }
        }

        let box_index = Self::calculate_box_index(position);
        for position in BoxIter::new(box_index) {
            let cell = self.board[position];
            if let Cell::Marked(marks) = cell {
                if marks.is_empty() {
                    return Err(BoardNotSolvableError);
                }

                if let Some(num) = marks.naked_single() {
                    self.insert_and_forward_propagate(num, position, Origin::ForwardPropagate)?;
                }
            }
        }

        Ok(())
    }

    fn maybe_update_mcv_candidate(&mut self, proposed_len: u8, position: BoardPosition) {
        let candidate_len = self.mcv_candidate.0;
        if proposed_len < candidate_len {
            self.mcv_candidate = (proposed_len, Some(position));
        }
    }

    fn clear_hidden_pairs_cache(&mut self) {
        for (_, positions) in self.hidden_pairs_cache.entries_mut() {
            positions.clear();
        }
    }

    fn insert_pair_combination_positions_into_cache<I>(&mut self, iter: I)
    where
        I: Iterator<Item = BoardPosition>,
    {
        for position in iter {
            let cell = self.board[position];
            if let Cell::Marked(marks) = cell {
                for combination in marks.combinations(2) {
                    unsafe {
                        self.hidden_pairs_cache
                            .get_unchecked_mut(combination.into())
                            .push(position);
                    }
                }
            }
        }
    }

    fn insert_constraints_from_hidden_pairs(&mut self, histogram: &Histogram) -> bool {
        let mut made_progress = false;

        let hidden_pairs = &mut self.hidden_pairs_cache;
        let constraints = self.constraint_stack.last_mut().expect("constraint stack is empty");

        for (marks, occurrences) in hidden_pairs.entries_mut() {
            if occurrences.len() == 2 {
                let occures_exactly_2_times = marks
                    .into_iter()
                    .map(|number| (number as usize) - 1)
                    .fold(true, |acc, num_index| acc & (histogram[num_index] == 2));

                if occures_exactly_2_times {
                    self.board[occurrences[0]] = Cell::Marked(marks);
                    self.board[occurrences[1]] = Cell::Marked(marks);

                    made_progress = true;

                    let marks = {
                        let mut it = marks.into_iter();
                        let m0 = it.next().unwrap();
                        let m1 = it.next().unwrap();
                        (m0, m1)
                    };

                    constraints.push(Constraint::NakedPair {
                        marks,
                        positions: (occurrences[0], occurrences[1]),
                    });
                }
            }
        }

        made_progress
    }

    fn compute_constraint_histogram_iter<I>(&self, iter: I) -> Histogram
    where
        I: Iterator<Item = BoardPosition>,
    {
        let mut histogram = [0; MAX_NUMBER_COUNT];
        for position in iter {
            let cell = self.board[position];
            if let Cell::Marked(marks) = cell {
                for number in marks {
                    let num_index = (number as usize) - 1;
                    histogram[num_index] += 1;
                }
            }
        }
        histogram
    }

    fn insert_hidden_pairs(&mut self) {
        let mut made_progress = false;
        for i in 0..9 {
            {
                self.clear_hidden_pairs_cache();
                self.insert_pair_combination_positions_into_cache(RowIter::new(i));
                let histogram = self.compute_constraint_histogram_iter(RowIter::new(i));

                made_progress |= self.insert_constraints_from_hidden_pairs(&histogram);
            }

            {
                self.clear_hidden_pairs_cache();
                self.insert_pair_combination_positions_into_cache(ColIter::new(i));
                let histogram = self.compute_constraint_histogram_iter(ColIter::new(i));

                made_progress |= self.insert_constraints_from_hidden_pairs(&histogram);
            }

            {
                self.clear_hidden_pairs_cache();
                self.insert_pair_combination_positions_into_cache(BoxIter::new(i));
                let histogram = self.compute_constraint_histogram_iter(BoxIter::new(i));

                made_progress |= self.insert_constraints_from_hidden_pairs(&histogram);
            }
        }
        self.tier = Tier::Tier2 {
            made_progress,
            ran_last_time: true,
        };
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Tier {
    Tier1 { made_progress: bool },
    Tier2 { made_progress: bool, ran_last_time: bool },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum HiddenSingleEntry {
    None,
    One(BoardPosition),
    Many,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum Constraint {
    NakedPair {
        marks: (SudokuNum, SudokuNum),
        positions: (BoardPosition, BoardPosition),
    },
}

pub mod tracing {
    use super::{Board, BoardPosition, SudokuNum};

    #[derive(Debug)]
    pub struct Trace {
        // NOTE(Simon): This could would be cleaner if root was of type (normal) `Board`
        // NOTE(Simon): but in order to keep duplicate code in the tree visualizer to a minimum
        // NOTE(Simon): we are using a `BoardWithConstraints` filled with just empty constraints
        pub root: Board,
        pub events: Vec<Event>,
    }

    #[derive(Debug)]
    pub enum Event {
        Insert {
            origin: Origin,

            position: BoardPosition,
            number: SudokuNum,

            board: Board,
        },
        PartiallyPropagate {
            board: Board,
        },
        Restore,
        Solved {
            board: Board,
        },
    }

    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
    pub enum Origin {
        Unspecified,
        ForwardPropagate,
        NakedSingle,
        HiddenSingle,
        DFSTryAll,
        DFSMCV,
        DFSConstraints,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::board::parse_board;
    use crate::visualize::*;
    use crate::*;

    use std::assert_matches::assert_matches;

    #[test]
    fn calculate_box_position() {
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
                let position = BigBoardPosition::new(row_index, col_index);
                let box_position = Solver::calculate_box_index(position);
                assert!(box_position < 9);

                let expected_box_position = expected[row_index][col_index];
                assert_eq!(box_position, expected_box_position);
            }
        }
    }

    #[test]
    fn insert_and_forward_propagate() {
        let board = parse_board(vec![
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
        let mut solver = Solver::new(board);
        let position = BigBoardPosition::new(8, 0);

        solver.update_initial_constraints();
        solver.partially_propagate_constraints();
        solver
            .insert_and_forward_propagate(SudokuNum::Three, position, Origin::Unspecified)
            .unwrap();

        let row_index = position.row_index;
        let col_index = position.col_index;
        let box_index = Solver::calculate_box_index(position);

        assert_eq!(solver.board[position], Cell::Number(SudokuNum::Three));

        for position in RowIter::new(row_index) {
            let cell = board[position];
            if let Cell::Marked(marks) = cell {
                assert_eq!(marks.contains(SudokuNum::Three), false);
            }
        }

        for position in ColIter::new(col_index) {
            let cell = board[position];
            if let Cell::Marked(marks) = cell {
                assert_eq!(marks.contains(SudokuNum::Three), false);
            }
        }

        for position in BoxIter::new(box_index) {
            let cell = board[position];
            if let Cell::Marked(marks) = cell {
                assert_eq!(marks.contains(SudokuNum::Three), false);
            }
        }
    }

    #[test]
    fn solve_board_1() {
        let board = parse_board(vec![
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

        let board_solution = parse_board(vec![
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
        assert!(board_solution.is_solved());

        let mut solver = Solver::new(board);
        let res = solver.solve();

        dbg!(&solver.board);

        assert_matches!(res, Ok(b) if b.is_solved() && b == board_solution);

        #[cfg(feature = "visualize")]
        visualize_trace(&solver.trace, "render/board_1.gv");
    }

    #[test]
    fn solve_board_4() {
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

        #[cfg(feature = "visualize")]
        visualize_trace(&solver.trace, "render/board_4.gv");

        assert_matches!(res, Ok(b) if b.is_solved() && b == board_4_solution)
    }

    #[test]
    fn solve_hard_leetcode() {
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
        assert_matches!(res, Ok(b) if b.is_solved() && b == hard_leetcode_solution);
    }
}
