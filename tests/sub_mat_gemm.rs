//! Test that:
//!
//! - `(alpha * A * B - beta * C)[r, c] == alpha * A[r, :] * B[:, c] - beta * C[r, c]`
//! - `(beta * C - alpha * A * B)[r, c] == beta * C[r, c] - alpha * A[r, :] * B[:, c]`
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

// alpha * A * B - beta * C
// beta * C - alpha * A * B
mod nnn {
    use onezero::Zero;
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
                    reverse: bool,
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

                    let _0 = $ty::zero();
                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);

                    if reverse {
                        let c = alpha * a * b - beta * c;

                        test_approx_eq! {
                            c[(row, col)],
                            alpha * dot - beta * e
                        }
                    } else {
                        let c = beta * c - alpha * a * b;

                        test_approx_eq! {
                            c[(row, col)],
                            beta * e - alpha * dot
                        }
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A * B - beta * C^t
// beta * C^t - alpha * A * B
mod nnt {
    use onezero::Zero;
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
                    reverse: bool,
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

                    let _0 = $ty::zero();
                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);

                    if reverse {
                        let c = alpha * a * b - beta * c;

                        test_approx_eq! {
                            c[(row, col)],
                            alpha * dot - beta * e
                        }
                    } else {
                        let c = beta * c - alpha * a * b;

                        test_approx_eq! {
                            c[(row, col)],
                            beta * e - alpha * dot
                        }
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A * B^t - beta * C
// beta * C - alpha * A * B^t
mod ntn {
    use onezero::Zero;
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
                    reverse: bool,
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

                    let _0 = $ty::zero();
                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);

                    if reverse {
                        let c = alpha * a * b - beta * c;

                        test_approx_eq! {
                            c[(row, col)],
                            alpha * dot - beta * e
                        }
                    } else {
                        let c = beta * c - alpha * a * b;

                        test_approx_eq! {
                            c[(row, col)],
                            beta * e - alpha * dot
                        }
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A * B^t - beta * C^t
// beta * C^t - alpha * A * B^t
mod ntt {
    use onezero::Zero;
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
                    reverse: bool,
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

                    let _0 = $ty::zero();
                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);

                    if reverse {
                        let c = alpha * a * b - beta * c;

                        test_approx_eq! {
                            c[(row, col)],
                            alpha * dot - beta * e
                        }
                    } else {
                        let c = beta * c - alpha * a * b;

                        test_approx_eq! {
                            c[(row, col)],
                            beta * e - alpha * dot
                        }
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A^t * B - beta * C
// beta * C - alpha * A^t * B
mod tnn {
    use onezero::Zero;
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
                    reverse: bool,
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

                    let _0 = $ty::zero();
                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);

                    if reverse {
                        let c = alpha * a * b - beta * c;

                        test_approx_eq! {
                            c[(row, col)],
                            alpha * dot - beta * e
                        }
                    } else {
                        let c = beta * c - alpha * a * b;

                        test_approx_eq! {
                            c[(row, col)],
                            beta * e - alpha * dot
                        }
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A^t * B - beta * C^t
// beta * C^t - alpha * A^t * B
mod tnt {
    use onezero::Zero;
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
                    reverse: bool,
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

                    let _0 = $ty::zero();
                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);

                    if reverse {
                        let c = alpha * a * b - beta * c;

                        test_approx_eq! {
                            c[(row, col)],
                            alpha * dot - beta * e
                        }
                    } else {
                        let c = beta * c - alpha * a * b;

                        test_approx_eq! {
                            c[(row, col)],
                            beta * e - alpha * dot
                        }
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A^t * B^t - beta * C
// beta * C - alpha * A^t * B^t
mod ttn {
    use onezero::Zero;
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
                    reverse: bool,
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

                    let _0 = $ty::zero();
                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);

                    if reverse {
                        let c = alpha * a * b - beta * c;

                        test_approx_eq! {
                            c[(row, col)],
                            alpha * dot - beta * e
                        }
                    } else {
                        let c = beta * c - alpha * a * b;

                        test_approx_eq! {
                            c[(row, col)],
                            beta * e - alpha * dot
                        }
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// alpha * A^t * B^t - beta * C^t
// beta * C^t - alpha * A^t * B^t
mod ttt {
    use onezero::Zero;
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
                    reverse: bool,
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

                    let _0 = $ty::zero();
                    let e = c[(row, col)];
                    let dot = a.row(row) * b.col(col);

                    if reverse {
                        let c = alpha * a * b - beta * c;

                        test_approx_eq! {
                            c[(row, col)],
                            alpha * dot - beta * e
                        }
                    } else {
                        let c = beta * c - alpha * a * b;

                        test_approx_eq! {
                            c[(row, col)],
                            beta * e - alpha * dot
                        }
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
