//! Given:
//!
//! `y <- alpha * A * x`
//!
//! Test that
//!
//! `y[i] = alpha * A[i, :] * x`
//!
//! for any valid `i`

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate approx;
extern crate complex;
extern crate linalg;
extern crate quickcheck;
extern crate rand;

#[macro_use]
mod setup;

use complex::{c64, c128};
use linalg::prelude::*;
use quickcheck::TestResult;

mod transposed {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), i: u32) -> TestResult {
                    enforce! {
                        i < nrows,
                        ncols != 0,
                    }

                    let a = ::setup::rand::mat((srow + ncols, scol + nrows));
                    let a = a.slice((srow.., scol..)).t();
                    let ref x = ::setup::rand::col(ncols);
                    let alpha: $t = ::setup::rand::scalar();
                    let mut y = ::setup::rand::col(nrows);

                    y.set(alpha * a * x);

                    test_approx_eq!(y[i], alpha * a.row(i) * x)
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);

}

macro_rules! tests {
    ($($t:ident),+) => {
        $(
            #[quickcheck]
            fn $t((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), i: u32) -> TestResult {
                enforce! {
                    i < nrows,
                    ncols != 0,
                }

                let a = ::setup::rand::mat((srow + nrows, scol + ncols));
                let a = a.slice((srow.., scol..));
                let ref x = ::setup::rand::col(ncols);
                let alpha: $t = ::setup::rand::scalar();
                let mut y = ::setup::rand::col(nrows);

                y.set(alpha * a * x);

                test_approx_eq!(y[i], alpha * a.row(i) * x)
            }
         )+
    }
}

tests!(f32, f64, c64, c128);
