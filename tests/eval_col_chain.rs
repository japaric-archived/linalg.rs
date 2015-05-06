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
                    (m, i, j): (u32, u32, u32),
                    row: u32 ,
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        row < m,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let a = ::setup::rand::mat((m, i));
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::col(j);

                    let z = (alpha * &a * &b * &c).eval();
                    let ab = (&a * &b).eval();

                    let dot = ab.row(row) * &c;

                    test_approx_eq! {
                        alpha * dot,
                        z[row]
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
                    (m, i, j, k): (u32, u32, u32, u32),
                    row: u32 ,
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        k != 0,
                        row < m,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let a = ::setup::rand::mat((m, i));
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, k));
                    let d = ::setup::rand::col(k);

                    let z = (alpha * &a * &b * &c * &d).eval();
                    let abc = (&a * &b * &c).eval();

                    let dot = abc.row(row) * &d;

                    test_approx_eq! {
                        alpha * dot,
                        z[row]
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
                    (m, i, j, k, l): (u32, u32, u32, u32, u32),
                    row: u32 ,
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        k != 0,
                        l != 0,
                        row < m,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let a = ::setup::rand::mat((m, i));
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, k));
                    let d = ::setup::rand::mat((k, l));
                    let e = ::setup::rand::col(l);

                    let z = (alpha * &a * &b * &c * &d * &e).eval();
                    let abc = (&a * &b * &c * &d).eval();

                    let dot = abc.row(row) * &e;

                    test_approx_eq! {
                        alpha * dot,
                        z[row]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
