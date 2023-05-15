#![feature(let_chains)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
// #![warn(clippy::restriction)]

use cli_table::{Style, Table};
use rustc_hash::{FxHashMap, FxHasher};
use std::fmt;
use std::hash::{Hash, Hasher};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SudokuNum {
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum NewCell {
    Number(SudokuNum),
    Constrained([Option<SudokuNum>; 9]),
    Free,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Cell {
    Number(u8),
    Constrained(Vec<u8>),
    Free,
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::Constrained(c) => write!(f, "{c:?}"),
            Self::Free => write!(f, "."),
        }
    }
}

type Board = [[Cell; 9]; 9];

#[derive(Debug, Copy, Clone)]
struct BoardNotSolvableError;

impl fmt::Display for BoardNotSolvableError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "failed to solve board")
    }
}

fn parse_board(board: Vec<Vec<char>>) -> Board {
    let mut new_board = std::array::from_fn(|_| std::array::from_fn(|_| Cell::Free));

    for row_index in 0..9 {
        for col_index in 0..9 {
            let char_cell = board[row_index][col_index];
            let cell = match char_cell {
                '0'..='9' => Cell::Number(char_cell.to_digit(10).unwrap() as u8),
                '.' => Cell::Free,
                _ => panic!("invalid char"),
            };
            new_board[row_index][col_index] = cell;
        }
    }

    new_board
}

fn to_solution(board: Board) -> Vec<Vec<char>> {
    let mut new_board = vec![];
    for row in board {
        let mut new_row = vec![];
        for cell in row {
            let ch = match cell {
                Cell::Number(n) => n as char,
                Cell::Free => '.',
                Cell::Constrained(_) => panic!(),
            };
            new_row.push(ch);
        }
        new_board.push(new_row);
    }
    new_board
}

fn print_board(board: &Board) {
    let table = board.table().bold(true).display().unwrap();

    println!(
        "\n{table}\n",
        // termion::clear::All,
        // termion::cursor::Goto(1, 1)
    );
}

fn insert_initial_constraints(board: &mut Board) {
    for row_index in 0..9 {
        let mut possible_nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        for col_index in 0..9 {
            let cell = &board[row_index][col_index];
            if let Cell::Number(n) = &cell {
                if let Some(index) = possible_nums.iter().position(|x| *x == *n) {
                    possible_nums.remove(index);
                }
            }
        }
        for col_index in 0..9 {
            let cell = &mut board[row_index][col_index];
            if let Cell::Free = &cell {
                *cell = Cell::Constrained(possible_nums.clone());
            }
        }
    }
}

// propagate constraints
fn partially_propagate_constraints(board: &mut Board) {
    fn partially_propagate_row_constraints(board: &mut Board) {
        for row_index in 0..9 {
            let mut found_nums = Vec::with_capacity(9);
            for col_index in 0..9 {
                let cell = &board[row_index][col_index];
                if let Cell::Number(n) = cell {
                    found_nums.push(*n);
                }
            }
            for col_index in 0..9 {
                let cell = &mut board[row_index][col_index];
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
                if let Cell::Number(n) = board[row_index][col_index] {
                    found_nums.push(n);
                }
            }
            for row_index in 0..9 {
                let cell = &mut board[row_index][col_index];
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
                if let Cell::Number(n) = board[row_index][col_index] {
                    found_nums.push(n);
                }
            }

            for (offset_y, offset_x) in OFFSETS {
                let row_index = (square_row_index + offset_y) as usize;
                let col_index = (square_col_index + offset_x) as usize;
                if let Cell::Constrained(ref mut constraints) = &mut board[row_index][col_index] {
                    for num in &found_nums {
                        if let Some(index) = constraints.iter().position(|x| x == num) {
                            constraints.remove(index);
                        }
                    }
                }
            }
        }
    }
    partially_propagate_row_constraints(board);
    partially_propagate_col_constraints(board);

    partially_propagate_square_constraints(board);
}

fn solve_board(board: &mut Board) -> SolveStatusProgress {
    fn solve_board_internal(board: &mut Board, is_first_iteration: bool) -> SolveStatusProgress {
        fn solve_board_dfs(board: &mut Board) -> SolveStatusProgress {
            for row_index in 0..9 {
                for col_index in 0..9 {
                    let cell = board[row_index][col_index].clone();
                    if let Cell::Constrained(cons) = cell {
                        for c in cons {
                            let mut new_board = board.clone();
                            new_board[row_index][col_index] = Cell::Number(c);
                            // print_board(&new_board);
                            if let SolveStatusProgress::Solved = solve_board_internal(&mut new_board, false) {
                                *board = new_board;
                                return SolveStatusProgress::Solved;
                            }
                        }
                    }
                }
            }
            SolveStatusProgress::NotSolvable
        }

        if is_first_iteration {
            insert_initial_constraints(board);
        }
        while !board_is_solved(&board) {
            partially_propagate_constraints(board);
            let status = insert_forced_constraints(board);
            match status {
                SolveStatusProgress::Stalling => return solve_board_dfs(board),
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
    solve_board_internal(board, true)
}

fn board_is_solved(board: &Board) -> bool {
    for row in board {
        for cell in row {
            if let Cell::Constrained(_) | Cell::Free = cell {
                return false;
            }
        }
    }

    // check horizontal rows
    for row in board {
        let mut required_nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        for cell in row {
            if let Cell::Number(n) = cell {
                if let Some(index) = required_nums.iter().position(|x| x == n) {
                    required_nums.remove(index);
                }
            }
        }
        if !required_nums.is_empty() {
            return false;
        }
    }

    // check vertical columns
    for col_index in 0..9 {
        let mut required_nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        for row_index in 0..9 {
            let cell = &board[row_index][col_index];
            if let Cell::Number(n) = cell {
                if let Some(index) = required_nums.iter().position(|x| x == n) {
                    required_nums.remove(index);
                }
            }
        }
        if !required_nums.is_empty() {
            return false;
        }
    }

    for (square_row_index, square_col_index) in MIDDLE_OF_SQUARE_INDEXES {
        let mut required_nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        for (offset_y, offset_x) in OFFSETS {
            let row_index = (square_row_index + offset_y) as usize;
            let col_index = (square_col_index + offset_x) as usize;

            let cell = &board[row_index][col_index];
            if let Cell::Number(n) = cell {
                if let Some(index) = required_nums.iter().position(|x| x == n) {
                    required_nums.remove(index);
                }
            }
        }
        if !required_nums.is_empty() {
            return false;
        }
    }

    true
}

#[derive(Debug, Copy, Clone)]
enum SolveStatusProgress {
    Solved,
    MadeProgress,
    NotSolvable,
    Stalling,
}

fn insert_forced_constraints(board: &mut Board) -> SolveStatusProgress {
    fn insert_obviously_forced_constraints(board: &mut Board) -> SolveStatusProgress {
        // a `forced` constraint is a constraint with len == 1
        for row in board.iter_mut() {
            for cell in row {
                match cell {
                    // puzzle not solvable
                    Cell::Constrained(cons) if cons.is_empty() => {
                        // print_board(board);
                        return SolveStatusProgress::NotSolvable;
                    }
                    Cell::Constrained(cons) if cons.len() == 1 => {
                        *cell = Cell::Number(cons.last().cloned().unwrap());
                    }
                    _ => continue,
                }
            }
        }
        SolveStatusProgress::MadeProgress
    }

    fn insert_forced_constraints_in_col(board: &mut Board) {
        for col_index in 0..9 {
            // occurences number of occurence and indexes
            let mut occurrences: FxHashMap<u8, Vec<(usize, usize)>> = FxHashMap::default();
            for row_index in 0..9 {
                let cell = &board[row_index][col_index];
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
                        board[*row_index][*col_index] = Cell::Number(i);
                    }
                }
            }
        }
    }

    fn insert_forced_constraints_in_row(board: &mut Board) {
        for col_index in 0..9 {
            // occurences number of occurence and indexes
            let mut occurrences: FxHashMap<u8, Vec<(usize, usize)>> = FxHashMap::default();
            for row_index in 0..9 {
                let cell = &board[row_index][col_index];
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
                        board[*row_index][*col_index] = Cell::Number(i);
                    }
                }
            }
        }
    }

    fn insert_forced_constraints_in_squares(board: &mut Board) {
        for (square_row_index, square_col_index) in MIDDLE_OF_SQUARE_INDEXES {
            let mut occurrences = FxHashMap::default();
            for (offset_y, offset_x) in OFFSETS {
                let row_index = (square_row_index + offset_y) as usize;
                let col_index = (square_col_index + offset_x) as usize;

                let cell = &board[row_index][col_index];
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
                        board[*row_index][*col_index] = Cell::Number(i);
                    }
                }
            }
        }
    }

    let old_board_hash = {
        let mut hasher = FxHasher::default();
        Hash::hash(board, &mut hasher);
        hasher.finish()
    };

    if let SolveStatusProgress::NotSolvable = insert_obviously_forced_constraints(board) {
        return SolveStatusProgress::NotSolvable;
    }
    insert_forced_constraints_in_row(board);
    insert_forced_constraints_in_col(board);
    insert_forced_constraints_in_squares(board);

    let new_board_hash = {
        let mut hasher = FxHasher::default();
        Hash::hash(board, &mut hasher);
        hasher.finish()
    };

    if old_board_hash == new_board_hash {
        return SolveStatusProgress::Stalling;
    }
    SolveStatusProgress::MadeProgress
}

fn main() -> Result<(), BoardNotSolvableError> {
    println!("{}", std::mem::size_of::<NewCell>());
    // let mut really_hard_test = parse_board(vec![
    //     vec!['.', '.', '.', '.', '7', '.', '1', '.', '.'],
    //     vec!['.', '.', '.', '5', '6', '.', '.', '.', '.'],
    //     vec!['.', '8', '.', '.', '2', '.', '.', '3', '.'],
    //     vec!['.', '.', '.', '.', '.', '.', '.', '4', '9'],
    //     vec!['.', '4', '.', '2', '5', '.', '.', '.', '8'],
    //     vec!['5', '.', '.', '9', '.', '.', '.', '.', '6'],
    //     vec!['4', '6', '.', '.', '.', '.', '2', '8', '.'],
    //     vec!['2', '.', '.', '.', '.', '.', '.', '.', '.'],
    //     vec!['7', '.', '.', '1', '9', '.', '8', '.', '.'],
    // ]);

    let mut codegolf = parse_board(vec![
        vec!['.', '.', '.', '7', '.', '.', '.', '.', '.'],
        vec!['1', '.', '.', '.', '.', '.', '.', '.', '.'],
        vec!['.', '.', '.', '4', '3', '.', '2', '.', '.'],
        vec!['.', '.', '.', '.', '.', '.', '.', '.', '6'],
        vec!['.', '.', '.', '5', '.', '9', '.', '.', '.'],
        vec!['.', '.', '.', '.', '.', '.', '4', '1', '8'],
        vec!['.', '.', '.', '.', '8', '1', '.', '.', '.'],
        vec!['.', '.', '2', '.', '.', '.', '.', '5', '.'],
        vec!['.', '4', '.', '.', '.', '.', '3', '.', '.'],
    ]);

    let mut hard_leetcode = parse_board(vec![
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

    let mut test_board = parse_board(vec![
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

    let now = std::time::Instant::now();
    let status = solve_board(&mut codegolf);

    println!("status = {:?} :: time = {:?}", status, now.elapsed());

    print_board(&codegolf);
    Ok(())
}
