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
extern crate onezero;
extern crate quickcheck;
extern crate rand;

#[macro_use]
mod setup;

mod _3 {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use onezero::Zero;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (m, i, j, n): (u32, u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let a = ::setup::rand::mat((m, i));
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, n));

                    let z = (alpha * &a * &b * &c).eval();
                    let ab = (&a * &b).eval();

                    let _0 = $ty::zero();
                    let dot = ab.row(row) * c.col(col);

                    test_approx_eq! {
                        alpha * dot,
                        z[(row, col)]
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
    use onezero::Zero;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (m, i, j, k, n): (u32, u32, u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let a = ::setup::rand::mat((m, i));
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, k));
                    let d = ::setup::rand::mat((k, n));

                    let z = (alpha * &a * &b * &c * &d).eval();
                    let ab = (&a * &b).eval();
                    let cd = (&c * &d).eval();

                    let _0 = $ty::zero();
                    let dot = ab.row(row) * cd.col(col);

                    test_approx_eq! {
                        alpha * dot,
                        z[(row, col)]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

mod _5 {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use onezero::Zero;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (m, i, j, k, l, n): (u32, u32, u32, u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        k != 0,
                        l != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let a = ::setup::rand::mat((m, i));
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, k));
                    let d = ::setup::rand::mat((k, l));
                    let e = ::setup::rand::mat((l, n));

                    let z = (alpha * &a * &b * &c * &d * &e).eval();
                    let abc = (&a * &b * &c).eval();
                    let de = (&d * &e).eval();

                    let _0 = $ty::zero();
                    let dot = abc.row(row) * de.col(col);

                    test_approx_eq! {
                        alpha * dot,
                        z[(row, col)]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
