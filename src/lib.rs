#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(internal_features)]
#![allow(dead_code)]
#![allow(incomplete_features)]
#![feature(rustc_attrs)]
#![feature(associated_type_bounds)]
#![feature(assert_matches)]
#![feature(test)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(generic_const_exprs)]
#![feature(let_chains)]

pub mod board;
pub mod generated_lut;
pub mod solver;
pub mod subset_cache;
mod visualize;
