use std::io::Write;
use std::process::Command;

pub struct CombinationsIter {
    k: u32,
    bits: usize,
    current: usize,
}

impl Iterator for CombinationsIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current <= self.bits {
            if self.current & self.bits == self.current && self.current.count_ones() == self.k {
                let result = self.current;
                let tmp = self.current & (!self.current + 1);
                let mobile = self.current + tmp;
                self.current = (((mobile ^ self.current) >> 2) / tmp) | mobile;
                return Some(result);
            }
            let tmp = self.current & (!self.current + 1);
            let mobile = self.current + tmp;
            self.current = (((mobile ^ self.current) >> 2) / tmp) | mobile;
        }
        None
    }
}

fn generate_combinations_lookup_table((lo, hi): (usize, usize)) -> std::io::Result<()> {
    let mut luts = vec![];
    for k in 2..=4 {
        let mut bins = vec![];
        for i in lo..=hi {
            let it = CombinationsIter {
                k,
                bits: i,
                current: (1 << k) - 1,
            };
            let combinations: Vec<usize> = it.into_iter().collect();
            if combinations.is_empty() {
                continue;
            }
            bins.push(combinations);
        }
        luts.push(bins);
    }
    let mut f = std::fs::File::create("src/generated_lut.rs")?;
    writeln!(f, "use crate::board::PencilMarks; \n")?;

    writeln!(f, "#[allow(clippy::unreadable_literal)]")?;
    write!(f, "const COMBINATIONS: [&[&[PencilMarks]]; 3] = [")?;
    for lut in luts {
        writeln!(f, "&[").unwrap();
        for bin in lut {
            let nums = bin
                .iter()
                .map(|n| format!("PencilMarks::from_raw_bits(0b{n:09b})"))
                .collect::<Vec<String>>()
                .join(", ");
            write!(f, "&[{nums}], ")?;
        }
        writeln!(f, "], ")?;
    }
    write!(f, "];")?;

    Ok(())
}

fn main() {
    println!("cargo:rerun_if_changed=build.rs");
    generate_combinations_lookup_table((0, 512)).expect("failed to generate lookup file");

    Command::new("cargo")
        .arg("fmt")
        .output()
        .expect("Failed to run cargo fmt");
}
