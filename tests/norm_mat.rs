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

mod transposed {
    use complex::{Math, c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t((nrows, ncols): (u32, u32)) -> TestResult {
                    let a = ::setup::rand::mat::<$t>((nrows, ncols)).t();

                    let norm = a.norm();
                    let ssq = a.iter().map(|&x| x * x).sum();

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
                fn $t((nrows, ncols): (u32, u32)) -> TestResult {
                    let a = ::setup::rand::mat::<$t>((nrows, ncols)).t();

                    let norm = a.norm();
                    let ssq = a.iter().map(|&x| x.abs() * x.abs()).sum();

                    test_approx_eq!(norm * norm, ssq)
                }
             )+
        }
    }

    complex_tests!(c64, c128);
}

macro_rules! tests {
    ($($t:ident),+) => {
        $(
            #[quickcheck]
            fn $t((nrows, ncols): (u32, u32)) -> TestResult {
                let a = ::setup::rand::mat::<$t>((nrows, ncols));

                let norm = a.norm();
                let ssq = a.iter().map(|&x| x * x).sum();

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
            fn $t((nrows, ncols): (u32, u32)) -> TestResult {
                let a = ::setup::rand::mat::<$t>((nrows, ncols));

                let norm = a.norm();
                let ssq = a.iter().map(|&x| x.abs() * x.abs()).sum();

                test_approx_eq!(norm * norm, ssq)
            }
         )+
    }
}

complex_tests!(c64, c128);
