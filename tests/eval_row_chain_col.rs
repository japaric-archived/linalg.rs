//! Test that:
//!
//! `(alpha * A * B * C * D) == alpha * (A * B) * (C * D)`
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
                    (i, j, k): (u32, u32, u32),
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        k != 0,
                    }

                    let a = ::setup::rand::row::<$ty>(i);
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, k));
                    let d = ::setup::rand::col(k);

                    let z = &a * &b * &c * &d;
                    let ab = (&a * &b).eval();
                    let cd = (&c * &d).eval();

                    let dot = &ab * &cd;

                    test_approx_eq! {
                        dot,
                        z
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
                    (i, j, k, l): (u32, u32, u32, u32),
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        k != 0,
                        l != 0,
                    }

                    let a = ::setup::rand::row::<$ty>(i);
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, k));
                    let d = ::setup::rand::mat((k, l));
                    let e = ::setup::rand::col(l);

                    let z = &a * &b * &c * &d * &e;
                    let abc = (&a * &b * &c).eval();
                    let de = (&d * &e).eval();

                    let dot = &abc * &de;

                    test_approx_eq! {
                        dot,
                        z
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
                    (i, j, k, l, m): (u32, u32, u32, u32, u32),
                ) -> TestResult {
                    enforce! {
                        i != 0,
                        j != 0,
                        k != 0,
                        l != 0,
                        m != 0,
                    }

                    let a = ::setup::rand::row::<$ty>(i);
                    let b = ::setup::rand::mat((i, j));
                    let c = ::setup::rand::mat((j, k));
                    let d = ::setup::rand::mat((k, l));
                    let e = ::setup::rand::mat((l, m));
                    let f = ::setup::rand::col(m);

                    let z = &a * &b * &c * &d * &e * &f;
                    let abc = (&a * &b * &c).eval();
                    let def = (&d * &e * &f).eval();

                    let dot = &abc * &def;

                    test_approx_eq! {
                        dot,
                        z
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
