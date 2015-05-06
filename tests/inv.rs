//! Test that:
//!
//! - `A[r, :] * (A^-1)[:, c] == if r == c { 1 } else { 0 }`
//! - `(A^-1)[r, :] * A[:, c] == if r == c { 1 } else { 0 }`

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
use onezero::{One, Zero};
use quickcheck::TestResult;

mod transposed {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use onezero::{One, Zero};
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(n: u32, (row, col): (u32, u32)) -> TestResult {
                    enforce! {
                        row < n,
                        col < n,
                    }

                    let _0 = $t::zero();
                    let _1 = $t::one();
                    let a = ::setup::rand::mat((n, n)).t();
                    let a_inv = a.clone().inv();

                    if row == col {
                        test_approx_eq!(a.row(row) * a_inv.col(col), _1);
                        test_approx_eq!(a_inv.row(row) * a.col(col), _1)
                    } else {
                        test_approx_eq!(a.row(row) * a_inv.col(col), _0);
                        test_approx_eq!(a_inv.row(row) * a.col(col), _0)
                    }
                }
             )+
        };
    }

    tests!(f32, f64, c64, c128);
}

macro_rules! tests {
    ($($t:ident),+) => {
        $(
            #[quickcheck]
            fn $t(n: u32, (row, col): (u32, u32)) -> TestResult {
                enforce! {
                    row < n,
                    col < n,
                }

                let _0 = $t::zero();
                let _1 = $t::one();
                let a = ::setup::rand::mat((n, n));
                let a_inv = a.clone().inv();

                if row == col {
                    test_approx_eq!(a.row(row) * a_inv.col(col), _1);
                    test_approx_eq!(a_inv.row(row) * a.col(col), _1)
                } else {
                    test_approx_eq!(a.row(row) * a_inv.col(col), _0);
                    test_approx_eq!(a_inv.row(row) * a.col(col), _0)
                }
            }
         )+
    };
}

tests!(f32, f64, c64, c128);
