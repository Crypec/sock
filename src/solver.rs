use rustc_hash::FxHashMap;
use strum::EnumCount;

use crate::board::{
    BigBoardPosition, Board, BoardIter, BoardWithConstraints, BoxIter, Cell, CellWithConstraints, ColIter,
    ConstraintList, RowIter, SudokuNum,
};

use tracing::*;

type BoardPosition = BigBoardPosition;

pub const MAX_NUMBER_COUNT: usize = SudokuNum::COUNT;

#[derive(Debug, Copy, Clone)]
pub struct BoardNotSolvableError;

type HiddenSinglesEntryList = [HiddenSingleEntry; MAX_NUMBER_COUNT];
type MissingNumbersList = [ConstraintList; MAX_NUMBER_COUNT];

pub struct Solver {
    pub board: Board,

    hidden_sets_row_cache: FxHashMap<ConstraintList, Vec<BoardPosition>>,
    hidden_sets_col_cache: FxHashMap<ConstraintList, Vec<BoardPosition>>,
    hidden_sets_box_cache: FxHashMap<ConstraintList, Vec<BoardPosition>>,

    made_progress: bool,

    // most constraining value
    mcv_candidate: (u8, Option<BoardPosition>),

    col_missing: MissingNumbersList,
    row_missing: MissingNumbersList,
    box_missing: MissingNumbersList,

    hidden_singles_row_cache: HiddenSinglesEntryList,
    hidden_singles_col_cache: HiddenSinglesEntryList,
    hidden_singles_box_cache: HiddenSinglesEntryList,

    #[cfg(feature = "tracing")]
    pub trace: Trace,
}

impl Solver {
    // because the are only 9 possible numbers in sudoku
    const MCV_CANDIDATE_MAX_LEN: u8 = MAX_NUMBER_COUNT as u8;

    pub fn new(board: Board) -> Self {
        Self {
            board,

            hidden_sets_row_cache: FxHashMap::default(),
            hidden_sets_col_cache: FxHashMap::default(),
            hidden_sets_box_cache: FxHashMap::default(),

            mcv_candidate: (Self::MCV_CANDIDATE_MAX_LEN, None),

            made_progress: false,

            col_missing: [ConstraintList::full(); MAX_NUMBER_COUNT],
            row_missing: [ConstraintList::full(); MAX_NUMBER_COUNT],
            box_missing: [ConstraintList::full(); MAX_NUMBER_COUNT],

            hidden_singles_row_cache: [HiddenSingleEntry::None; MAX_NUMBER_COUNT],
            hidden_singles_col_cache: [HiddenSingleEntry::None; MAX_NUMBER_COUNT],
            hidden_singles_box_cache: [HiddenSingleEntry::None; MAX_NUMBER_COUNT],

            #[cfg(feature = "tracing")]
            trace: Trace {
                root: None,
                events: vec![],
            },
        }
    }

    pub fn solve(&mut self) -> Result<Board, BoardNotSolvableError> {
        #[cfg(feature = "tracing")]
        {
            let mut constrained_board = self.get_constrained_board();

            // HACK(Simon): Because we start with a full `ConstraintList` we need to clear all constraints
            for pos in BoardIter::new() {
                let cell = &mut constrained_board.0[pos.row_index][pos.col_index];
                if let CellWithConstraints::Constrained(cons) = cell {
                    *cons = ConstraintList::empty();
                }
            }
            self.trace.root = Some(constrained_board);
        }

        self.partially_propagate_constraints();

        #[cfg(feature = "tracing")]
        {
            let constrained_board = self.get_constrained_board();
            self.trace.events.push(Event::PartiallyPropagate {
                board: constrained_board,
            });
        }

        self.solve_internal(0)
    }

    fn solve_internal(&mut self, depth: usize) -> Result<Board, BoardNotSolvableError> {
        while !self.is_solved() && self.made_progress {
            self.made_progress = false;
            self.insert_naked_singles()?;
            self.insert_hidden_singles()?;
            // self.compute_hidden_subsets();
        }
        if self.is_solved() {
            return Ok(self.board);
        }
        self.solve_dfs(depth)
    }

    fn is_solved(&self) -> bool {
        if self.row_missing.iter().any(|c| !c.is_empty())
            || self.col_missing.iter().any(|c| !c.is_empty())
            || self.box_missing.iter().any(|c| !c.is_empty())
        {
            return false;
        }
        self.board.is_solved()
    }

    // If we have exhausted all our options using constraint propagation, run a depth first search
    fn solve_dfs(&mut self, depth: usize) -> Result<Board, BoardNotSolvableError> {
        // println!("{depth}");
        if let Some(position) = self.mcv_candidate.1 {
            self.invalidate_mcv_candidate();

            let cell = self.board[position];
            if cell == Cell::Free {
                let cons = self.constraints_at(position);

                for c in cons {
                    let old_row_missing = self.row_missing;
                    let old_col_missing = self.col_missing;
                    let old_box_missing = self.box_missing;

                    let old_board = self.board;

                    self.remove_cons_at_pos(c, position);

                    if self.insert_and_forward_propagate(c, position, Origin::DFS).is_err() {
                        self.restore_state(old_board, old_row_missing, old_col_missing, old_box_missing);
                        continue;
                    }

                    match self.solve_internal(depth + 1) {
                        Ok(board) => {
                            #[cfg(feature = "tracing")]
                            {
                                let solved_board = self.get_constrained_board();
                                self.trace.events.push(Event::Solved { board: solved_board });
                            }
                            return Ok(board);
                        }
                        Err(BoardNotSolvableError) => {
                            self.restore_state(old_board, old_row_missing, old_col_missing, old_box_missing);
                        }
                    }
                }
                return Err(BoardNotSolvableError);
            }
        }

        // if we don't have a candidate for the mcv heuristic we do a linear search for the next free cell
        for position in BoardIter::new() {
            let cell = self.board[position];
            if cell == Cell::Free {
                let cons = self.constraints_at(position);

                for c in cons {
                    let old_row_missing = self.row_missing;
                    let old_col_missing = self.col_missing;
                    let old_box_missing = self.box_missing;

                    let old_board = self.board;

                    if self.insert_and_forward_propagate(c, position, Origin::DFS).is_err() {
                        self.restore_state(old_board, old_row_missing, old_col_missing, old_box_missing);
                        continue;
                    }

                    match self.solve_internal(depth + 1) {
                        Ok(board) => {
                            #[cfg(feature = "tracing")]
                            {
                                let solved_board = self.get_constrained_board();
                                self.trace.events.push(Event::Solved { board: solved_board });
                            }

                            return Ok(board);
                        }
                        Err(BoardNotSolvableError) => {
                            self.restore_state(old_board, old_row_missing, old_col_missing, old_box_missing);
                        }
                    }
                }
                return Err(BoardNotSolvableError);
            }
        }
        Err(BoardNotSolvableError)
    }

    #[inline(always)]
    fn restore_state(
        &mut self,
        old_board: Board,
        old_row_missing: MissingNumbersList,
        old_col_missing: MissingNumbersList,
        old_box_missing: MissingNumbersList,
    ) {
        #[cfg(feature = "tracing")]
        self.trace.events.push(Event::Restore);

        self.board = old_board;
        self.row_missing = old_row_missing;
        self.col_missing = old_col_missing;
        self.box_missing = old_box_missing;
    }

    // fn test(&mut self) {
    //     for i in 0..9 {
    //         let mut hidden_singles_row_cache = self.hidden_singles_row_cache;
    //         self.compute_hidden_singles_cache_iter(&mut hidden_singles_row_cache, RowIter::new(1));
    //         self.insert_hidden_singles_row_entries();
    //     }
    // }

    // fn compute_hidden_singles_cache_iter<Iter>(
    //     &self,
    //     hidden_singles_cache_entry_list: &mut HiddenSinglesEntryList,
    //     iter: Iter,
    // ) where
    //     Iter: Iterator<Item = (usize, usize)>,
    // {
    //     *hidden_singles_cache_entry_list = [HiddenSingleEntry::None; 9];
    //     for (row_index, col_index) in iter {
    //         let cell = self.board.0[row_index][col_index];
    //         if cell == Cell::Free {
    //             let cons = self.constraints_at(row_index, col_index);

    //             for sudoku_num in cons {
    //                 Self::maybe_update_hidden_singles_entry(
    //                     hidden_singles_cache_entry_list,
    //                     sudoku_num,
    //                     row_index,
    //                     col_index,
    //                 );
    //             }
    //         }
    //     }
    // }

    fn insert_hidden_singles(&mut self) -> Result<(), BoardNotSolvableError> {
        for i in 0..9 {
            self.hidden_singles_row_cache = [HiddenSingleEntry::None; 9];
            for position in RowIter::new(i) {
                let cell = self.board[position];

                if cell == Cell::Free {
                    let cons = self.constraints_at(position);
                    for sudoku_num in cons {
                        Self::maybe_update_hidden_singles_entry(
                            &mut self.hidden_singles_row_cache,
                            sudoku_num,
                            position,
                        );
                    }
                }
            }
            self.insert_hidden_singles_row_entries()?;

            self.hidden_singles_col_cache = [HiddenSingleEntry::None; 9];
            for position in ColIter::new(i) {
                let cell = self.board[position];
                if cell == Cell::Free {
                    let cons = self.constraints_at(position);
                    for sudoku_num in cons {
                        Self::maybe_update_hidden_singles_entry(
                            &mut self.hidden_singles_col_cache,
                            sudoku_num,
                            position,
                        );
                    }
                }
            }
            self.insert_hidden_singles_col_entries()?;

            self.hidden_singles_box_cache = [HiddenSingleEntry::None; 9];
            for position in BoxIter::new(i) {
                let cell = self.board[position];
                if cell == Cell::Free {
                    let cons = self.constraints_at(position);
                    for sudoku_num in cons {
                        Self::maybe_update_hidden_singles_entry(
                            &mut self.hidden_singles_box_cache,
                            sudoku_num,
                            position,
                        );
                    }
                }
            }
            self.insert_hidden_singles_box_entries()?;
        }
        Ok(())
    }

    fn insert_hidden_singles_row_entries(&mut self) -> Result<(), BoardNotSolvableError> {
        unsafe {
            let this = self as *mut Self;
            for (num_position, entry) in self.hidden_singles_row_cache.iter().enumerate() {
                if let HiddenSingleEntry::One(position) = entry {
                    let num: SudokuNum = (num_position + 1)
                        .try_into()
                        .expect("failed to convert to sudoku number");
                    (*this).insert_and_forward_propagate(num, *position, Origin::HiddenSingle)?;
                }
            }
        }
        Ok(())
    }

    fn insert_hidden_singles_col_entries(&mut self) -> Result<(), BoardNotSolvableError> {
        unsafe {
            let this = self as *mut Self;
            for (num_position, entry) in self.hidden_singles_col_cache.iter().enumerate() {
                if let HiddenSingleEntry::One(position) = entry {
                    let num: SudokuNum = (num_position + 1)
                        .try_into()
                        .expect("failed to convert to sudoku number");
                    (*this).insert_and_forward_propagate(num, *position, Origin::HiddenSingle)?;
                }
            }
        }
        Ok(())
    }

    fn insert_hidden_singles_box_entries(&mut self) -> Result<(), BoardNotSolvableError> {
        unsafe {
            let this = self as *mut Self;
            for (num_position, entry) in self.hidden_singles_box_cache.iter().enumerate() {
                if let HiddenSingleEntry::One(position) = entry {
                    let num: SudokuNum = (num_position + 1)
                        .try_into()
                        .expect("failed to convert to sudoku number");

                    (*this).insert_and_forward_propagate(num, *position, Origin::HiddenSingle)?;
                }
            }
        }
        Ok(())
    }

    fn maybe_update_hidden_singles_entry(
        hidden_singles_entry_list: &mut HiddenSinglesEntryList,
        num: SudokuNum,
        position: BoardPosition,
    ) {
        let num_position = (num as usize) - 1;
        let old_entry = hidden_singles_entry_list[num_position];
        let new_entry = match old_entry {
            HiddenSingleEntry::None => HiddenSingleEntry::One(position),
            HiddenSingleEntry::One(..) | HiddenSingleEntry::Many => HiddenSingleEntry::Many,
        };
        hidden_singles_entry_list[num_position] = new_entry;
    }

    fn invalidate_mcv_candidate(&mut self) {
        self.mcv_candidate.0 = Self::MCV_CANDIDATE_MAX_LEN;
        self.mcv_candidate.1 = None;
    }

    const fn constraints_at(&self, position: BoardPosition) -> ConstraintList {
        let row_index = position.row_index;
        let col_index = position.col_index;

        let box_position = Self::calculate_box_position(position);

        let row_missing = self.row_missing[row_index];
        let col_missing = self.col_missing[col_index];
        let box_missing = self.box_missing[box_position];

        ConstraintList::intersection(row_missing, col_missing, box_missing)
    }

    fn insert_naked_singles(&mut self) -> Result<(), BoardNotSolvableError> {
        for position in BoardIter::new() {
            let cell = self.board[position];

            if cell == Cell::Free {
                let cons = self.constraints_at(position);
                if cons.is_empty() {
                    return Err(BoardNotSolvableError);
                } else if let Some(num) = cons.naked_single() {
                    self.insert_and_forward_propagate(num, position, Origin::NakedSingle)?;
                }
            }
        }
        Ok(())
    }

    fn clear_row_subset_cache(&mut self) {
        for positiones in &mut self.hidden_sets_row_cache.values_mut() {
            positiones.clear();
        }
    }

    fn clear_col_subset_cache(&mut self) {
        for positiones in &mut self.hidden_sets_col_cache.values_mut() {
            positiones.clear();
        }
    }

    fn clear_box_cache(&mut self) {
        for positiones in &mut self.hidden_sets_box_cache.values_mut() {
            positiones.clear();
        }
    }

    fn compute_hidden_subsets(&mut self) -> Result<(), BoardNotSolvableError> {
        for i in 0..9 {
            // hidden subsets in row
            {
                self.clear_row_subset_cache();
                for position in RowIter::new(i) {
                    let cell = self.board[position];

                    if cell == Cell::Free {
                        let cons = self.constraints_at(position);
                        for k in 2..=4 {
                            let subsets = cons.combinations(k);
                            for s in subsets {
                                let occurrences = self
                                    .hidden_sets_row_cache
                                    .entry(s)
                                    .or_insert_with(|| Vec::with_capacity(9));
                                occurrences.push(position);
                            }
                        }
                    }
                }
                self.insert_constraints_from_hidden_sets_row_cache()?;
            }

            // hidden singles in columns
            {
                self.clear_col_subset_cache();
                for position in ColIter::new(i) {
                    let cell = self.board[position];

                    if cell == Cell::Free {
                        let cons = self.constraints_at(position);
                        for k in 2..=4 as u8 {
                            let subsets = cons.combinations(k);
                            for s in subsets {
                                let occurrences = self
                                    .hidden_sets_col_cache
                                    .entry(s)
                                    .or_insert_with(|| Vec::with_capacity(9));
                                occurrences.push(position);
                            }
                        }
                    }
                }
                self.insert_constraints_from_hidden_sets_col_cache()?;
            }

            // hidden singles in boxes
            {
                self.clear_box_cache();
                for position in BoxIter::new(i) {
                    let cell = self.board[position];
                    if cell == Cell::Free {
                        let cons = self.constraints_at(position);
                        for k in 2..=4 as u8 {
                            let subsets = cons.combinations(k);
                            for s in subsets {
                                let occurrences = self
                                    .hidden_sets_box_cache
                                    .entry(s)
                                    .or_insert_with(|| Vec::with_capacity(9));
                                occurrences.push(position);
                            }
                        }
                    }
                }
                self.insert_constraints_from_hidden_sets_box_cache()?;
            }
        }
        Ok(())
    }

    fn insert_constraints_from_hidden_sets_row_cache(&mut self) -> Result<(), BoardNotSolvableError> {
        unsafe {
            let this = self as *mut Self;
            Self::insert_constraints_from_hidden_sets_cache_raw(this, &self.hidden_sets_row_cache)?;
        }
        Ok(())
    }

    fn insert_constraints_from_hidden_sets_col_cache(&mut self) -> Result<(), BoardNotSolvableError> {
        unsafe {
            let this = self as *mut Self;
            Self::insert_constraints_from_hidden_sets_cache_raw(this, &self.hidden_sets_col_cache)?;
        }

        Ok(())
    }

    fn insert_constraints_from_hidden_sets_box_cache(&mut self) -> Result<(), BoardNotSolvableError> {
        unsafe {
            let this = self as *mut Self;
            Self::insert_constraints_from_hidden_sets_cache_raw(this, &self.hidden_sets_box_cache)?;
        }
        Ok(())
    }

    unsafe fn insert_constraints_from_hidden_sets_cache_raw(
        this: *mut Self,
        hidden_sets_cache: &FxHashMap<ConstraintList, Vec<BoardPosition>>,
    ) -> Result<(), BoardNotSolvableError> {
        for (cons, occurrences) in hidden_sets_cache {
            // we have found a subset of length `k` that is occurring exactly `k` times
            if cons.len() > 1 && cons.len() as usize == occurrences.len() {
                for _position in occurrences {
                    // also remove from all the boxes
                    // we do this here because every cell from the occurrence list could be in different squares
                    // (*this).dbg_missing();
                    // (*this).remove_all_cons_at_pos(*row_index, *col_index, *cons);
                    // (*this).insert_cons_at_pos(*row_index, *col_index, *cons);
                    // println!("we fucked up");
                }
            }
        }

        // PERF(Simon): Maybe it is beneficial to first insert all hidden subsets with 2 =< k =< 4.
        // PERF(Simon): This could allow us to find more hidden singles while forward propagation due to a reduced search space.
        for (cons, occurrences) in hidden_sets_cache {
            // hidden singles
            if let Some(num) = cons.naked_single() && occurrences.len() == 1 {
                let position = occurrences[0];
                (*this).insert_and_forward_propagate(num, position, Origin::Unspecified)?;
            }
        }

        Ok(())
    }

    const fn calculate_box_position(position: BoardPosition) -> usize {
        let row_index = position.row_index;
        let col_index = position.col_index;

        (row_index / 3) * 3 + col_index / 3
    }

    // fn remove_all_cons_at_pos(&mut self, row_index: usize, col_index: usize, to_remove: ConstraintList) {
    //     let square_position = Self::calculate_square_position(row_index, col_index);

    //     self.row_missing[row_index].remove_all(to_remove);
    //     self.col_missing[col_index].remove_all(to_remove);
    //     self.square_missing[square_position].remove_all(to_remove);
    // }

    fn remove_cons_at_pos(&mut self, to_remove: SudokuNum, position: BoardPosition) {
        let row_index = position.row_index;
        let col_index = position.col_index;
        let box_position = Self::calculate_box_position(position);

        self.row_missing[row_index].remove(to_remove);
        self.col_missing[col_index].remove(to_remove);
        self.box_missing[box_position].remove(to_remove);
    }

    // propagate constraints
    pub fn partially_propagate_constraints(&mut self) {
        self.partially_propagate_row_constraints();
        self.partially_propagate_col_constraints();
        self.partially_propagate_box_constraints();
    }

    fn partially_propagate_row_constraints(&mut self) {
        for row_index in 0..9 {
            let mut found_nums = ConstraintList::empty();
            for position in RowIter::new(row_index) {
                let cell = self.board[position];
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
            for position in ColIter::new(col_index) {
                let cell = self.board[position];
                if let Cell::Number(n) = cell {
                    found_nums.insert(n);
                }
            }
            self.col_missing[col_index].remove_all(found_nums);
        }
    }
    fn partially_propagate_box_constraints(&mut self) {
        for box_position in 0..9 {
            let mut found_nums = ConstraintList::empty();
            for position in BoxIter::new(box_position) {
                let cell = self.board[position];
                if let Cell::Number(n) = cell {
                    found_nums.insert(n);
                }
            }
            self.box_missing[box_position].remove_all(found_nums);
        }
    }

    fn insert_and_forward_propagate(
        &mut self,
        number: SudokuNum,
        position: BoardPosition,
        _origin: Origin,
    ) -> Result<(), BoardNotSolvableError> {
        #[cfg(debug_assertions)]
        {
            use std::assert_matches::*;
            let cons = self.constraints_at(position);
            // dbg!(cons);
            // dbg!(self.get_constrained_board());
            debug_assert!(
                cons.contains(number),
                "the placement of a number has to be at least partially valid :: {} : {:?}",
                number,
                (position.row_index, position.col_index),
            );
            let cell = self.board[position];

            // dbg!(&self.hidden_sets_square_cache);
            // dbg!((row_index, col_index, num));
            debug_assert_matches!(cell, Cell::Free if cons.contains(number), "the placement of a number has to be at least partially valid :: tried to insert {} into :: {:?}", number, (position.row_index, position.col_index));
        }

        #[cfg(feature = "tracing")]
        {
            let board = self.get_constrained_board();

            self.trace.events.push(Event::Insert {
                origin: _origin,
                position,
                number,
                board,
            });
        }

        // actually insert the number into the board
        self.board[position] = Cell::Number(number);
        self.made_progress = true;

        self.remove_cons_at_pos(number, position);

        // PERF(Simon): Maybe we can fuse these loops to avoid doing duplicate iteration over the same data
        // PERF(Simon): I'm not sure the compiler is smart enough to do this optimization on it's own.
        for position in RowIter::new(position.row_index) {
            let cell = self.board[position];
            if cell == Cell::Free {
                let cons = self.constraints_at(position);

                if cons.is_empty() {
                    return Err(BoardNotSolvableError);
                }

                // check if we maybe found a more constraining candidate
                self.maybe_update_mcv_candidate(cons.len() as u8, position);

                if let Some(num) = cons.naked_single() {
                    self.insert_and_forward_propagate(num, position, Origin::ForwardPropagate)?;
                }
            }
        }

        for position in ColIter::new(position.col_index) {
            let cell = self.board[position];
            if cell == Cell::Free {
                let cons = self.constraints_at(position);

                if cons.is_empty() {
                    return Err(BoardNotSolvableError);
                }

                if let Some(num) = cons.naked_single() {
                    self.insert_and_forward_propagate(num, position, Origin::ForwardPropagate)?;
                }
            }
        }

        let box_index = Self::calculate_box_position(position);
        for position in BoxIter::new(box_index) {
            let cell = self.board[position];
            if cell == Cell::Free {
                let cons = self.constraints_at(position);

                if cons.is_empty() {
                    return Err(BoardNotSolvableError);
                }

                if let Some(num) = cons.naked_single() {
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

    pub fn is_subset(&self, other: &Board) -> bool {
        let mut is_subset = true;
        for position in BoardIter::new() {
            let cell = self.board[position];
            let needle = other[position];
            match (cell, needle) {
                (Cell::Number(n1), Cell::Number(n2)) if n1 == n2 => continue,
                (Cell::Free, Cell::Number(n)) => {
                    let cons = self.constraints_at(position);
                    if cons.contains(n) {
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
        for position in BoardIter::new() {
            let old_cell = self.board[position];
            let new_cell = match old_cell {
                Cell::Number(n) => CellWithConstraints::Number(n),
                Cell::Free => {
                    let cons = self.constraints_at(position);
                    CellWithConstraints::Constrained(cons)
                }
            };
            let row_index = position.row_index;
            let col_index = position.col_index;
            new_board.0[row_index][col_index] = new_cell;
        }
        new_board
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum HiddenSingleEntry {
    None,
    One(BoardPosition),
    Many,
}

pub mod tracing {
    use super::*;

    #[derive(Debug)]
    pub struct Trace {
        // NOTE(Simon): This could would be cleaner if root was of type (normal) `Board`
        // NOTE(Simon): but in order to keep duplicate code in the tree visualizer to a minimum
        // NOTE(Simon): we are using a `BoardWithConstraints` filled with just empty constraints
        pub root: Option<BoardWithConstraints>,
        pub events: Vec<Event>,
    }

    #[derive(Debug)]
    pub enum Event {
        Insert {
            origin: Origin,

            position: BoardPosition,
            number: SudokuNum,

            board: BoardWithConstraints,
        },
        PartiallyPropagate {
            board: BoardWithConstraints,
        },
        Restore,
        Solved {
            board: BoardWithConstraints,
        },
    }

    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
    pub enum Origin {
        Unspecified,
        ForwardPropagate,
        NakedSingle,
        HiddenSingle,
        DFS,
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
                let box_position = Solver::calculate_box_position(position);
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
        solver.partially_propagate_constraints();
        solver
            .insert_and_forward_propagate(SudokuNum::Three, BigBoardPosition::new(8, 0), Origin::Unspecified)
            .unwrap();

        assert_eq!(solver.board.0[8][0], Cell::Number(SudokuNum::Three));
        for position in RowIter::new(8) {
            let cons = solver.constraints_at(position);
            assert_eq!(cons.contains(SudokuNum::Three), false);
        }

        for position in ColIter::new(0) {
            let cons = solver.constraints_at(position);
            assert_eq!(cons.contains(SudokuNum::Three), false);
        }

        for position in BoxIter::new(6) {
            let cons = solver.constraints_at(position);
            assert_eq!(cons.contains(SudokuNum::Three), false);
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

        assert_matches!(res, Ok(b) if b.is_solved() && b == board_solution);
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
