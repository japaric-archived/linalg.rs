extern crate linalg;

use linalg::prelude::*;
use std::mem;

// Test that the return type of `At::at/AtMut::at_mut` is one-word long
#[test]
fn at() {
    let mut m = Mat::from_fn((2, 2), |i| i).unwrap();

    assert_eq!(mem::size_of_val(&m.at((0, 0))), mem::size_of::<&u8>());
    assert_eq!(mem::size_of_val(&m.at_mut((0, 0))), mem::size_of::<&u8>());
}
