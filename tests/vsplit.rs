//! Given:
//!
//! `(A, B) = C.vsplit_at[_mut](col)`
//!
//! Test that the iterators `A[row, :].iter()` and `B[row, :].iter()` are ordered and complete

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
        let (left, right) = v.vsplit_at(col);

        let mut i = 0;
        for x in left.row(row) {
            test_eq!(x, &(srow + i, scol + row));

            i += 1;
        }

        for x in right.row(row) {
            test_eq!(x, &(srow + i, scol + row));

            i += 1;
        }

        test_eq!(i, ncols)
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
        let (mut left, mut right) = v.vsplit_at_mut(col);

        let mut i = 0;
        for x in left.row_mut(row) {
            test_eq!(x, &mut (srow + i, scol + row));

            i += 1;
        }

        for x in right.row_mut(row) {
            test_eq!(x, &mut (srow + i, scol + row));

            i += 1;
        }

        test_eq!(i, ncols)
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
    let (left, right) = v.vsplit_at(col);

    let mut i = 0;
    for x in left.row(row) {
        test_eq!(x, &(srow + row, scol + i));

        i += 1;
    }

    for x in right.row(row) {
        test_eq!(x, &(srow + row, scol + i));

        i += 1;
    }

    test_eq!(i, ncols)
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
    let (mut left, mut right) = v.vsplit_at_mut(col);

    let mut i = 0;
    for x in left.row_mut(row) {
        test_eq!(x, &mut (srow + row, scol + i));

        i += 1;
    }

    for x in right.row_mut(row) {
        test_eq!(x, &mut (srow + row, scol + i));

        i += 1;
    }

    test_eq!(i, ncols)
}
