//! Test that:
//!
//! `(alpha * A * B * C * D)[r, c] == alpha * (A * B)[r, :] * (C * D)[:, c]`
//!
//! for any valid `r`, `c`

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

mod _2 {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (i, j, n): (u32, u32, u32),
                    col: u32 ,
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let a = ::setup::rand::row(i);
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, n));

                    let z = (alpha * &a * &b * &c).eval();
                    let bc = (&b * &c).eval();

                    let dot = &a * bc.col(col);

                    test_approx_eq! {
                        alpha * dot,
                        z[col]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

mod _3 {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (i, j, k, n): (u32, u32, u32, u32),
                    col: u32 ,
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        k != 0,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let a = ::setup::rand::row(i);
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, k));
                    let d = ::setup::rand::mat((k, n));

                    let z = (alpha * &a * &b * &c * &d).eval();
                    let bcd = (&b * &c * &d).eval();

                    let dot = &a * bcd.col(col);

                    test_approx_eq! {
                        alpha * dot,
                        z[col]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

mod _4 {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (i, j, k, l, n): (u32, u32, u32, u32, u32),
                    col: u32 ,
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        k != 0,
                        l != 0,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let a = ::setup::rand::row(i);
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, k));
                    let d = ::setup::rand::mat((k, l));
                    let e = ::setup::rand::mat((l, n));

                    let z = (alpha * &a * &b * &c * &d * &e).eval();
                    let bcde = (&b * &c * &d * &e).eval();

                    let dot = &a * bcde.col(col);

                    test_approx_eq! {
                        alpha * dot,
                        z[col]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
