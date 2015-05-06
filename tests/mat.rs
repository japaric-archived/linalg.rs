//! Test that
//!
//! - `fmt!("{:?}", mat)` == "[0, 1, 2]\n[3, 4, 5]", etc

#![feature(plugin)]
#![plugin(linalg_macros)]

extern crate linalg;
extern crate rand;

mod setup;

use linalg::prelude::*;

#[test]
fn col() {
    assert_eq![mat![0; 1; 2], (0..3).collect::<ColVec<_>>()];
}

#[test]
fn mat() {
    assert_eq!(mat![(0, 0), (0, 1); (1, 0), (1, 1)], Mat::from_fn((2, 2), |i| i));
}

#[test]
fn row() {
    assert_eq![mat![0, 1, 2], (0..3).collect::<RowVec<_>>()];
}

