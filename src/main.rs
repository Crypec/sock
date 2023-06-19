#![feature(let_chains)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
// #![warn(clippy::restriction)]

use crate::board::*;
use crate::solver::*;

mod board;
mod solver;

fn parse_boards_list(raw: String) -> Vec<Board> {
    raw.lines().skip(1).map(|line| parse_board_from_line(line)).collect()
}

fn parse_board_from_line(line: &str) -> Board {
    debug_assert_eq!(line.len(), 81);
    let mut new_board = std::array::from_fn(|_| std::array::from_fn(|_| Cell::Free));
    for row_index in 0..9 {
        for col_index in 0..9 {
            let char_cell = (line.as_bytes()[(row_index * 9) + col_index]) as char;
            let cell = match char_cell {
                '1'..='9' => Cell::Number(char_cell.to_digit(10).unwrap() as u8),
                '.' | '0' => Cell::Free,
                _ => panic!("invalid char"),
            };
            new_board[row_index][col_index] = cell;
        }
    }
    board::Board(new_board)
}

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

// #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
// enum SudokuNum {
//     One = 1,
//     Two = 2,
//     Three = 3,
//     Four = 4,
//     Five = 5,
//     Six = 6,
//     Seven = 7,
//     Eight = 8,
//     Nine = 9,
// }

// #[derive(Debug, Copy, Clone, PartialEq, Eq)]
// enum NewCell {
//     Number(SudokuNum),
//     Constrained([Option<SudokuNum>; 9]),
//     Free,
// }

fn main() -> Result<(), BoardNotSolvableError> {
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

    // let mut hard_leetcode = parse_board(vec![
    //     vec!['.', '.', '.', '.', '.', '7', '.', '.', '9'],
    //     vec!['.', '4', '.', '.', '8', '1', '2', '.', '.'],
    //     vec!['.', '.', '.', '9', '.', '.', '.', '1', '.'],
    //     vec!['.', '.', '5', '3', '.', '.', '.', '7', '2'],
    //     vec!['2', '9', '3', '.', '.', '.', '.', '5', '.'],
    //     vec!['.', '.', '.', '.', '.', '5', '3', '.', '.'],
    //     vec!['8', '.', '.', '.', '2', '3', '.', '.', '.'],
    //     vec!['7', '.', '.', '.', '5', '.', '.', '4', '.'],
    //     vec!['5', '3', '1', '.', '7', '.', '.', '.', '.'],
    // ]);

    // let mut test_board = parse_board(vec![
    //     vec!['5', '3', '.', '.', '7', '.', '.', '.', '.'],
    //     vec!['6', '.', '.', '1', '9', '5', '.', '.', '.'],
    //     vec!['.', '9', '8', '.', '.', '.', '.', '6', '.'],
    //     vec!['8', '.', '.', '.', '6', '.', '.', '.', '3'],
    //     vec!['4', '.', '.', '8', '.', '3', '.', '.', '1'],
    //     vec!['7', '.', '.', '.', '2', '.', '.', '.', '6'],
    //     vec!['.', '6', '.', '.', '.', '.', '2', '8', '.'],
    //     vec!['.', '.', '.', '4', '1', '9', '.', '.', '5'],
    //     vec!['.', '.', '.', '.', '8', '.', '.', '7', '9'],
    // ]);

    // let test_board_solution = parse_board(vec![
    //     vec!['5', '3', '4', '6', '7', '8', '9', '1', '2'],
    //     vec!['6', '7', '2', '1', '9', '5', '3', '4', '8'],
    //     vec!['1', '9', '8', '3', '4', '2', '5', '6', '7'],
    //     vec!['8', '5', '9', '7', '6', '1', '4', '2', '3'],
    //     vec!['4', '2', '6', '8', '5', '3', '7', '9', '1'],
    //     vec!['7', '1', '3', '9', '2', '4', '8', '5', '6'],
    //     vec!['9', '6', '1', '5', '3', '7', '2', '8', '4'],
    //     vec!['2', '8', '7', '4', '1', '9', '6', '3', '5'],
    //     vec!['3', '4', '5', '2', '8', '6', '1', '7', '9'],
    // ]);

    let test_data = std::fs::read_to_string("test_data.txt").unwrap();
    let mut boards = parse_boards_list(test_data);

    let now = std::time::Instant::now();
    for (index, board) in boards.into_iter().enumerate() {
        let now = std::time::Instant::now();
        let mut solver = Solver::new(board);
        let status = solver.solve();
        println!("solved board {index} :: in {:?}", now.elapsed());
    }

    println!("took :: {:?}", now.elapsed());

    // print_board(&mut boards[0]);
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn board_not_solved_empty_cells() {
        let codegolf = parse_board(vec![
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
        assert_eq!(board_is_solved(&codegolf), false);
    }
    #[test]
    fn board_not_solved_1() {
        let board = parse_board(vec![
            vec!['9', '3', '4', '6', '7', '8', '9', '1', '2'],
            vec!['6', '7', '2', '1', '9', '5', '3', '4', '8'],
            vec!['1', '9', '8', '3', '4', '2', '5', '6', '7'],
            vec!['8', '5', '9', '7', '6', '1', '4', '2', '3'],
            vec!['4', '2', '6', '8', '5', '3', '7', '9', '1'],
            vec!['7', '1', '3', '9', '2', '4', '8', '5', '6'],
            vec!['9', '6', '1', '5', '3', '7', '2', '8', '4'],
            vec!['2', '8', '7', '4', '1', '9', '6', '3', '5'],
            vec!['3', '4', '5', '2', '8', '6', '1', '7', '5'],
        ]);
        assert_eq!(board_is_solved(&board), false);
    }

    #[test]
    fn board_is_solved_2() {
        let board = parse_board(vec![
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
        assert_eq!(board_is_solved(&board), true);
    }
}
