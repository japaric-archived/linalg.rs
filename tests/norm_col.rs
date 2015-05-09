//! Test that:
//!
//! `norm(a)^2 == a.iter().map(|&x| x^2).sum()`

#![feature(core)]
#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate approx;
extern crate complex;
extern crate linalg;
extern crate quickcheck;
extern crate rand;

use complex::{Math, c64, c128};
use linalg::prelude::*;
use quickcheck::TestResult;

#[macro_use]
mod setup;

macro_rules! tests {
    ($($t:ident),+) => {
        $(
            #[quickcheck]
            fn $t((nrows, ncols): (u32, u32), col: u32) -> TestResult {
                enforce! {
                    col < ncols,
                }

                let a = ::setup::rand::mat::<$t>((nrows, ncols));
                let x = a.col(col);

                let norm = x.norm();
                let ssq = x.iter().map(|&x| x * x).sum();

                test_approx_eq!(norm * norm, ssq)
            }
         )+
    }
}

tests!(f32, f64);

macro_rules! complex_tests {
    ($($t:ident),+) => {
        $(
            #[quickcheck]
            fn $t((nrows, ncols): (u32, u32), col: u32) -> TestResult {
                enforce! {
                    col < ncols,
                }

                let a = ::setup::rand::mat::<$t>((nrows, ncols));
                let x = a.col(col);

                let norm = x.norm();
                let ssq = x.iter().map(|&x| x.abs() * x.abs()).sum();

                test_approx_eq!(norm * norm, ssq)
            }
         )+
    }
}

complex_tests!(c64, c128);

mod transposed {
    use complex::{Math, c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t((nrows, ncols): (u32, u32), col: u32) -> TestResult {
                    enforce! {
                        col < ncols,
                    }

                    let a = ::setup::rand::mat::<$t>((ncols, nrows)).t();
                    let x = a.col(col);

                    let norm = x.norm();
                    let ssq = x.iter().map(|&x| x * x).sum();

                    test_approx_eq!(norm * norm, ssq)
                }
             )+
        }
    }

    tests!(f32, f64);


    macro_rules! complex_tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t((nrows, ncols): (u32, u32), col: u32) -> TestResult {
                    enforce! {
                        col < ncols,
                    }

                    let a = ::setup::rand::mat::<$t>((ncols, nrows)).t();
                    let x = a.col(col);

                    let norm = x.norm();
                    let ssq = x.iter().map(|&x| x.abs() * x.abs()).sum();

                    test_approx_eq!(norm * norm, ssq)
                }
             )+
        }
    }

    complex_tests!(c64, c128);
}
