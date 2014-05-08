#![crate_id="linalg#0.11-pre"]
#![crate_type="lib"]
#![feature(macro_rules)]

extern crate num;
extern crate rand;

pub mod array;
pub mod blas;
pub mod mat;
pub mod vec;

mod traits;

#[cfg(test)]
mod bench;
mod test;
