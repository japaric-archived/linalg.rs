//! Given:
//!
//! `C = A .* B`
//!
//! Test that:
//!
//! `C[i, j] = A[i, j] * B[i, j]`
//!
//! for any valid `i`, `j`

#![feature(augmented_assignments)]
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

// A .* B
mod nn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (nrows, ncols): (u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < ncols,
                        row < nrows,
                    }

                    let mut a = ::setup::rand::mat::<$ty>((srow + nrows, scol + ncols));
                    let mut a = a.slice_mut((srow.., scol..));
                    let e = a[(row, col)];
                    let b = ::setup::rand::mat::<$ty>((srow + nrows, scol + ncols));
                    let b = b.slice((srow.., scol..));

                    a *= b;

                    test_approx_eq! {
                        a[(row, col)],
                        e * b[(row, col)]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// A .* B'
mod nt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (nrows, ncols): (u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < ncols,
                        row < nrows,
                    }

                    let mut a = ::setup::rand::mat::<$ty>((srow + nrows, scol + ncols));
                    let mut a = a.slice_mut((srow.., scol..));
                    let e = a[(row, col)];
                    let b = ::setup::rand::mat::<$ty>((srow + ncols, scol + nrows));
                    let b = b.slice((srow.., scol..)).t();

                    a *= b;

                    test_approx_eq! {
                        a[(row, col)],
                        e * b[(row, col)]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// A' .* B
mod tn {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (nrows, ncols): (u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < ncols,
                        row < nrows,
                    }

                    let mut a = ::setup::rand::mat::<$ty>((srow + ncols, scol + nrows));
                    let mut a = a.slice_mut((srow.., scol..)).t();
                    let e = a[(row, col)];
                    let b = ::setup::rand::mat::<$ty>((srow + nrows, scol + ncols));
                    let b = b.slice((srow.., scol..));

                    a *= b;

                    test_approx_eq! {
                        a[(row, col)],
                        e * b[(row, col)]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// A' .* B'
mod tt {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($ty:ident),+) => {
            $(
                #[quickcheck]
                fn $ty(
                    (srow, scol): (u32, u32),
                    (nrows, ncols): (u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < ncols,
                        row < nrows,
                    }

                    let mut a = ::setup::rand::mat::<$ty>((srow + ncols, scol + nrows));
                    let mut a = a.slice_mut((srow.., scol..)).t();
                    let e = a[(row, col)];
                    let b = ::setup::rand::mat::<$ty>((srow + ncols, scol + nrows));
                    let b = b.slice((srow.., scol..)).t();

                    a *= b;

                    test_approx_eq! {
                        a[(row, col)],
                        e * b[(row, col)]
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
