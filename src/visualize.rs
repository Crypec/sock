use crate::board::{BigBoardPosition, BoardWithConstraints, CellWithConstraints, SudokuNum};
use crate::solver::tracing::*;

pub fn visualize_trace<'p, P>(trace: &Trace, path: P)
where
    P: AsRef<std::path::Path>,
{
    dbg!(&trace.events);
    let root_board = emit_board(&trace.root.as_ref().expect("no root board set"), "root");

    let boards = trace
        .events
        .iter()
        .filter_map(|event| match event {
            Event::Insert {
                origin: _,
                position: _,
                number: _,
                board,
            }
            | Event::PartiallyPropagate { board }
            | Event::Solved { board } => Some(board),
            Event::Restore => None,
        })
        .enumerate()
        .map(|(i, board)| emit_board(&board, &format!("b_{i}")))
        .collect::<Vec<String>>()
        .join("\n\n\n");

    let edges = emit_edges(&trace.events);

    let graphviz_file_content = format!(
        r#"digraph SudokuBoard {{
        node [shape=plaintext];

        {root_board}        
        {boards}
        {edges}
}}"#
    );
    std::fs::write(path, graphviz_file_content).expect("failed to write graphviz file");
}

fn emit_edges(events: &[Event]) -> String {
    let mut stack = vec![];
    let mut edges = vec![];

    let mut prev_index = 0;
    let mut current_index = 0;

    // skip the first edge because we emit it as the root edge
    for event in events.iter().skip(1) {
        match event {
            Event::Insert {
                origin,
                position,
                number,
                board: _,
            } => {
                if *origin == Origin::DFS {
                    stack.push(current_index);
                }

                current_index += 1;

                let insert_label = emit_insert_label(&origin, &position, number);
                let edge = format!("b_{prev_index} -> b_{current_index} {insert_label}");

                prev_index = current_index;

                edges.push(edge);
            }
            Event::Restore => {
                prev_index = stack.pop().unwrap();
            }
            Event::PartiallyPropagate { board: _ } => {
                let edge = format!(r#"root -> b_0 [label = "partially propagate constraints"]"#);
                edges.push(edge);
            }
            Event::Solved { board: _ } => {}
        };
    }
    edges.join("\n")
}

fn emit_insert_label(origin: &Origin, index: &BigBoardPosition, number: &SudokuNum) -> String {
    let origin_text = match origin {
        Origin::Unspecified => "unspecified",
        Origin::NakedSingle => "naked single",
        Origin::HiddenSingle => "hidden single",
        Origin::ForwardPropagate => "forward propagation",
        Origin::DFS => "depth first search",
    };
    let index = format!("{:?}", (index.row_index, index.col_index));
    format!(r#"[ label ="{origin_text} :: {number} : {index}" ]"#)
}

fn emit_board(board: &BoardWithConstraints, name: &str) -> String {
    let board_table = emit_board_table(&board);
    format!("{name} [label=<\n    {board_table}\n        >];")
}

fn emit_board_table(board: &BoardWithConstraints) -> String {
    let trs = board
        .0
        .iter()
        .map(|row| emit_row(row))
        .collect::<Vec<String>>()
        .join("\n");
    format!(
        r#"        <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="20">
{trs}
            </TABLE>"#
    )
}

fn emit_row(row: &[CellWithConstraints]) -> String {
    let tds = row
        .iter()
        .map(|cell| match cell {
            CellWithConstraints::Number(num) => format!("                    <TD>{num}</TD>"),
            CellWithConstraints::Constrained(cons) => format!("                    <TD>{:?}</TD>", cons),
            CellWithConstraints::Free => format!("                    <TD></TD>"),
        })
        .collect::<Vec<String>>()
        .join("\n");
    format!("                <TR>\n{tds}\n                </TR>")
}
