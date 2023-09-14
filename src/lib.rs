#![feature(let_chains)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(internal_features)]
#![feature(rustc_attrs)]
#![feature(associated_type_bounds)]
#![feature(assert_matches)]
#![feature(test)]
#![allow(dead_code)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod board;
mod generated_lut;
pub mod solver;
pub mod subset_cache;
mod visualize;
