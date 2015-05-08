//! Given:
//!
//! `C <- alpha * A * B`
//!
//! Test that
//!
//! `C[i, j] = alpha * A[i, :] * B[:, j]`
//!
//! for any valid `i`

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

// C <- alpha * A * B
mod nnn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < n,
                        k != 0,
                        row < m,
                    }

                    let a = ::setup::rand::mat((srow + m, scol + k));
                    let a = a.slice((srow.., scol..));
                    let b = ::setup::rand::mat((srow + k, scol + n));
                    let b = b.slice((srow.., scol..));
                    let alpha: $t = ::setup::rand::scalar();
                    let mut c = ::setup::rand::mat((m, n));

                    c.set(alpha * a * b);

                    test_approx_eq!(c[(row, col)], alpha * a.row(row) * b.col(col))
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// C <- alpha * A * B'
mod nnt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < n,
                        k != 0,
                        row < m,
                    }

                    let a = ::setup::rand::mat((srow + m, scol + k));
                    let a = a.slice((srow.., scol..));
                    let b = ::setup::rand::mat((srow + n, scol + k));
                    let b = b.slice((srow.., scol..)).t();
                    let alpha: $t = ::setup::rand::scalar();
                    let mut c = ::setup::rand::mat((m, n));

                    c.set(alpha * a * b);

                    test_approx_eq!(c[(row, col)], alpha * a.row(row) * b.col(col))
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// C <- alpha * A' * B
mod ntn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < n,
                        k != 0,
                        row < m,
                    }

                    let a = ::setup::rand::mat((srow + k, scol + m));
                    let a = a.slice((srow.., scol..)).t();
                    let b = ::setup::rand::mat((srow + k, scol + n));
                    let b = b.slice((srow.., scol..));
                    let alpha: $t = ::setup::rand::scalar();
                    let mut c = ::setup::rand::mat((m, n));

                    c.set(alpha * a * b);

                    test_approx_eq!(c[(row, col)], alpha * a.row(row) * b.col(col))
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// C <- alpha * A' * B'
mod ntt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < n,
                        k != 0,
                        row < m,
                    }

                    let a = ::setup::rand::mat((srow + k, scol + m));
                    let a = a.slice((srow.., scol..)).t();
                    let b = ::setup::rand::mat((srow + n, scol + k));
                    let b = b.slice((srow.., scol..)).t();
                    let alpha: $t = ::setup::rand::scalar();
                    let mut c = ::setup::rand::mat((m, n));

                    c.set(alpha * a * b);

                    test_approx_eq!(c[(row, col)], alpha * a.row(row) * b.col(col))
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// C' <- alpha * A * B
mod tnn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < n,
                        k != 0,
                        row < m,
                    }

                    let a = ::setup::rand::mat((srow + m, scol + k));
                    let a = a.slice((srow.., scol..));
                    let b = ::setup::rand::mat((srow + k, scol + n));
                    let b = b.slice((srow.., scol..));
                    let alpha: $t = ::setup::rand::scalar();
                    let mut c = ::setup::rand::mat((n, m)).t();

                    c.set(alpha * a * b);

                    test_approx_eq!(c[(row, col)], alpha * a.row(row) * b.col(col))
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// C' <- alpha * A * B'
mod tnt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < n,
                        k != 0,
                        row < m,
                    }

                    let a = ::setup::rand::mat((srow + m, scol + k));
                    let a = a.slice((srow.., scol..));
                    let b = ::setup::rand::mat((srow + n, scol + k));
                    let b = b.slice((srow.., scol..)).t();
                    let alpha: $t = ::setup::rand::scalar();
                    let mut c = ::setup::rand::mat((n, m)).t();

                    c.set(alpha * a * b);

                    test_approx_eq!(c[(row, col)], alpha * a.row(row) * b.col(col))
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// C' <- alpha * A' * B
mod ttn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < n,
                        k != 0,
                        row < m,
                    }

                    let a = ::setup::rand::mat((srow + k, scol + m));
                    let a = a.slice((srow.., scol..)).t();
                    let b = ::setup::rand::mat((srow + k, scol + n));
                    let b = b.slice((srow.., scol..));
                    let alpha: $t = ::setup::rand::scalar();
                    let mut c = ::setup::rand::mat((n, m)).t();

                    c.set(alpha * a * b);

                    test_approx_eq!(c[(row, col)], alpha * a.row(row) * b.col(col))
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// C' <- alpha * A' * B'
mod ttt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (srow, scol): (u32, u32),
                    (m, k, n): (u32, u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < n,
                        k != 0,
                        row < m,
                    }

                    let a = ::setup::rand::mat((srow + k, scol + m));
                    let a = a.slice((srow.., scol..)).t();
                    let b = ::setup::rand::mat((srow + n, scol + k));
                    let b = b.slice((srow.., scol..)).t();
                    let alpha: $t = ::setup::rand::scalar();
                    let mut c = ::setup::rand::mat((n, m)).t();

                    c.set(alpha * a * b);

                    test_approx_eq!(c[(row, col)], alpha * a.row(row) * b.col(col))
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
