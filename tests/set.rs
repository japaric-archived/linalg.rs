//! Given:
//!
//! `A[r1, :] = B[r2, :]`
//!
//! Test that
//!
//! `A[r1, c] = B[r2, c]`
//!
//! for any valid `r1`, `r2`, `c`

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate complex;
extern crate linalg;
extern crate onezero;
extern crate quickcheck;
extern crate rand;

#[macro_use]
mod setup;

mod col {
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
                    (c1, c2, row): (u32, u32, u32),
                ) -> TestResult {
                    enforce! {
                        c1 < ncols,
                        c2 < ncols,
                        row < nrows,
                    }

                    let mut a = ::setup::rand::mat::<$t>((srow + nrows, scol + ncols));
                    let mut a = a.slice_mut((srow.., scol..));
                    let b = ::setup::rand::mat((srow + nrows, scol + ncols));
                    let b = b.slice((srow.., scol..));

                    a.col_mut(c1).set(b.col(c2));

                    test_eq!(a[(row, c1)], b[(row, c2)])
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

mod row {
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
                    (r1, r2, col): (u32, u32, u32),
                ) -> TestResult {
                    enforce! {
                        r1 < nrows,
                        r2 < nrows,
                        col < ncols,
                    }

                    let mut a = ::setup::rand::mat::<$t>((srow + nrows, scol + ncols));
                    let mut a = a.slice_mut((srow.., scol..));
                    let b = ::setup::rand::mat((srow + nrows, scol + ncols));
                    let b = b.slice((srow.., scol..));

                    a.row_mut(r1).set(b.row(r2));

                    test_eq!(a[(r1, col)], b[(r2, col)])
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

mod submat {
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

                    let mut a = ::setup::rand::mat::<$t>((srow + nrows, scol + ncols));
                    let mut a = a.slice_mut((srow.., scol..));
                    let b = ::setup::rand::mat((srow + nrows, scol + ncols));
                    let b = b.slice((srow.., scol..));
                    a.set(b);

                    test_eq!(a[(row, col)], b[(row, col)])
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
