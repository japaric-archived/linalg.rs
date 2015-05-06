//! Test that:
//!
//! `(alpha * A * B + beta * C)[r, c] == alpha * A[r, :] * B[:, c] + beta * C[r, c]`
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

// alpha * A * B + beta * C
mod nnn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let beta: $ty = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((srow + m, scol + k));
                    let a = a.slice((srow.., scol..));
                    let b = ::setup::rand::mat((srow + k, scol + n));
                    let b = b.slice((srow.., scol..));
                    let c = ::setup::rand::mat((m, n));

                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);
                    let c = alpha * a * b + beta * c;

                    test_approx_eq! {
                        c[(row, col)],
                        beta * e + alpha * dot
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A * B + beta * C^t
mod nnt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let beta: $ty = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((srow + m, scol + k));
                    let a = a.slice((srow.., scol..));
                    let b = ::setup::rand::mat((srow + k, scol + n));
                    let b = b.slice((srow.., scol..));
                    let c = ::setup::rand::mat((n, m)).t();

                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);
                    let c = alpha * a * b + beta * c;

                    test_approx_eq! {
                        c[(row, col)],
                        beta * e + alpha * dot
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A * B^t + beta * C
mod ntn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let beta: $ty = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((srow + m, scol + k));
                    let a = a.slice((srow.., scol..));
                    let b = ::setup::rand::mat((srow + n, scol + k));
                    let b = b.slice((srow.., scol..)).t();
                    let c = ::setup::rand::mat((m, n));

                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);
                    let c = alpha * a * b + beta * c;

                    test_approx_eq! {
                        c[(row, col)],
                        beta * e + alpha * dot
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A * B^t + beta * C^t
mod ntt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let beta: $ty = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((srow + m, scol + k));
                    let a = a.slice((srow.., scol..));
                    let b = ::setup::rand::mat((srow + n, scol + k));
                    let b = b.slice((srow.., scol..)).t();
                    let c = ::setup::rand::mat((n, m)).t();

                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);
                    let c = alpha * a * b + beta * c;

                    test_approx_eq! {
                        c[(row, col)],
                        beta * e + alpha * dot
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A^t * B + beta * C
mod tnn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let beta: $ty = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((srow + k, scol + m));
                    let a = a.slice((srow.., scol..)).t();
                    let b = ::setup::rand::mat((srow + k, scol + n));
                    let b = b.slice((srow.., scol..));
                    let c = ::setup::rand::mat((m, n));

                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);
                    let c = alpha * a * b + beta * c;

                    test_approx_eq! {
                        c[(row, col)],
                        beta * e + alpha * dot
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A^t * B + beta * C^t
mod tnt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let beta: $ty = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((srow + k, scol + m));
                    let a = a.slice((srow.., scol..)).t();
                    let b = ::setup::rand::mat((srow + k, scol + n));
                    let b = b.slice((srow.., scol..));
                    let c = ::setup::rand::mat((n, m)).t();

                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);
                    let c = alpha * a * b + beta * c;

                    test_approx_eq! {
                        c[(row, col)],
                        beta * e + alpha * dot
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A^t * B^t + beta * C
mod ttn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let beta: $ty = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((srow + k, scol + m));
                    let a = a.slice((srow.., scol..)).t();
                    let b = ::setup::rand::mat((srow + n, scol + k));
                    let b = b.slice((srow.., scol..)).t();
                    let c = ::setup::rand::mat((m, n));

                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);
                    let c = alpha * a * b + beta * c;

                    test_approx_eq! {
                        c[(row, col)],
                        beta * e + alpha * dot
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A^t * B^t + beta * C^t
mod ttt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let alpha: $ty = ::setup::rand::scalar();
                    let beta: $ty = ::setup::rand::scalar();

                    let a = ::setup::rand::mat((srow + k, scol + m));
                    let a = a.slice((srow.., scol..)).t();
                    let b = ::setup::rand::mat((srow + n, scol + k));
                    let b = b.slice((srow.., scol..)).t();
                    let c = ::setup::rand::mat((n, m)).t();

                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);
                    let c = alpha * a * b + beta * c;

                    test_approx_eq! {
                        c[(row, col)],
                        beta * e + alpha * dot
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
