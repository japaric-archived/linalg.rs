//! Given:
//!
//! `(A, B) = C.hsplit_at[_mut](row)`
//!
//! Test that the iterators `A[:, col].iter()` and `B[:, col].iter()` are ordered and complete for
//! any valid `col`

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate cast;
extern crate linalg;
extern crate quickcheck;
extern crate rand;

use linalg::prelude::*;
use quickcheck::TestResult;

#[macro_use]
mod setup;

mod transposed {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn submat(
        (srow, scol): (u32, u32),
        (nrows, ncols): (u32, u32),
        (row, col): (u32, u32),
    ) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        let m = ::setup::mat((srow + ncols, scol + nrows));
        let v = m.slice((srow.., scol..)).t();
        let (top, bottom) = v.hsplit_at(row);

        let mut i = 0;
        for x in top.col(col) {
            test_eq!(x, &(srow + col, scol + i));

            i += 1;
        }

        for x in bottom.col(col) {
            test_eq!(x, &(srow + col, scol + i));

            i += 1;
        }

        test_eq!(i, nrows)
    }

    #[quickcheck]
    fn submat_mut(
        (srow, scol): (u32, u32),
        (nrows, ncols): (u32, u32),
        (row, col): (u32, u32),
    ) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        let mut m = ::setup::mat((srow + ncols, scol + nrows));
        let mut v = m.slice_mut((srow.., scol..)).t();
        let (mut top, mut bottom) = v.hsplit_at_mut(row);

        let mut i = 0;
        for x in top.col_mut(col) {
            test_eq!(x, &mut (srow + col, scol + i));

            i += 1;
        }

        for x in bottom.col_mut(col) {
            test_eq!(x, &mut (srow + col, scol + i));

            i += 1;
        }

        test_eq!(i, nrows)
    }
}

#[quickcheck]
fn submat(
    (srow, scol): (u32, u32),
    (nrows, ncols): (u32, u32),
    (row, col): (u32, u32),
) -> TestResult {
    enforce! {
        row < nrows,
        col < ncols,
    }

    let m = ::setup::mat((srow + nrows, scol + ncols));
    let v = m.slice((srow.., scol..));
    let (top, bottom) = v.hsplit_at(row);

    let mut i = 0;
    for x in top.col(col) {
        test_eq!(x, &(srow + i, scol + col));

        i += 1;
    }

    for x in bottom.col(col) {
        test_eq!(x, &(srow + i, scol + col));

        i += 1;
    }

    test_eq!(i, nrows)
}

#[quickcheck]
fn submat_mut(
    (srow, scol): (u32, u32),
    (nrows, ncols): (u32, u32),
    (row, col): (u32, u32),
) -> TestResult {
    enforce! {
        row < nrows,
        col < ncols,
    }

    let mut m = ::setup::mat((srow + nrows, scol + ncols));
    let mut v = m.slice_mut((srow.., scol..));
    let (mut top, mut bottom) = v.hsplit_at_mut(row);

    let mut i = 0;
    for x in top.col_mut(col) {
        test_eq!(x, &mut (srow + i, scol + col));

        i += 1;
    }

    for x in bottom.col_mut(col) {
        test_eq!(x, &mut (srow + i, scol + col));

        i += 1;
    }

    test_eq!(i, nrows)
}
