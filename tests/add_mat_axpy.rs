//! Test that:
//!
//! `(A + alpha * B)[r, c] == A[r, c] + alpha * B[r, c]`
//!
//! for any valid `r`, `c`

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate onezero;
extern crate approx;
extern crate complex;
extern crate linalg;
extern crate quickcheck;
extern crate rand;

#[macro_use]
mod setup;

// A + alpha * B
mod nn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (nrows, ncols): (u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        col < ncols,
                    }

                    let alpha: $t = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((nrows, ncols));
                    let b = ::setup::rand::mat((srow + nrows, scol + ncols));
                    let b = b.slice((srow.., scol..));

                    let e = a[(row, col)];
                    let c = a + alpha * b;

                    test_approx_eq! {
                        c[(row, col)],
                        e + alpha * b[(row, col)]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// A + alpha * B^t
mod nt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (nrows, ncols): (u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        col < ncols,
                    }

                    let alpha: $t = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((nrows, ncols));
                    let b = ::setup::rand::mat((srow + ncols, scol + nrows));
                    let b = b.slice((srow.., scol..)).t();

                    let e = a[(row, col)];
                    let c = a + alpha * b;

                    test_approx_eq! {
                        c[(row, col)],
                        e + alpha * b[(row, col)]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// A^t + alpha * B
mod tn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (nrows, ncols): (u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        col < ncols,
                    }

                    let alpha: $t = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((ncols, nrows)).t();
                    let b = ::setup::rand::mat((srow + nrows, scol + ncols));
                    let b = b.slice((srow.., scol..));

                    let e = a[(row, col)];
                    let c = a + alpha * b;

                    test_approx_eq! {
                        c[(row, col)],
                        e + alpha * b[(row, col)]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// A^t + alpha * B^t
mod tt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (nrows, ncols): (u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        col < ncols,
                    }

                    let alpha: $t = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((ncols, nrows)).t();
                    let b = ::setup::rand::mat((srow + ncols, scol + nrows));
                    let b = b.slice((srow.., scol..)).t();

                    let e = a[(row, col)];
                    let c = a + alpha * b;

                    test_approx_eq! {
                        c[(row, col)],
                        e + alpha * b[(row, col)]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
