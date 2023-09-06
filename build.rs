use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=./test_data/");

    // build tdudoku solver

    let test_files_path = std::fs::read_dir("./test_data/").expect("failed to find test_data path");
    for path in test_files_path {
        let path = path.expect("failed to get path").path();
        let file_stem = path.file_stem().expect("failed to get file stem");
        println!("{file_stem:?}");
        // println!("Name: {}", path.unwrap().path().display())
    }
    panic!("stop");
}

fn write_test(path: String, name: &str, file: std::fs::File) {}
