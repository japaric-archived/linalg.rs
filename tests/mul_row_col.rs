//! Test that:
//!
//! `(alpha * A * B)[r, c] == alpha * A[r, :] * B[:, c]`
//!
//! for any valid `r`, `c`

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate approx;
extern crate complex;
extern crate linalg;
extern crate onezero;
extern crate quickcheck;
extern crate rand;

#[macro_use]
mod setup;

use complex::{c64, c128};
use linalg::prelude::*;
use onezero::Zero;
use quickcheck::TestResult;

macro_rules! tests {
    ($($ty:ident),+) => {
        $(
            #[quickcheck]
            fn $ty((m, k, n): (u32, u32, u32), (row, col): (u32, u32)) -> TestResult {
                enforce! {
                    k != 0,
                    row < m,
                    col < n,
                }

                let alpha: $ty = ::setup::rand::scalar();
                let ref a = ::setup::rand::mat((m, k));
                let ref b = ::setup::rand::mat((k, n));

                let dot = alpha * a.row(row) * b.col(col);
                let _0 = $ty::zero();
                let fold = a.row(row).iter().zip(b.col(col).iter()).fold(_0, |acc, (&x, &y)| {
                    acc + x * y
                });

                test_approx_eq! {
                    dot,
                    alpha * fold
                }
            }
         )+
    }
}

tests!(f32, f64, c64, c128);
