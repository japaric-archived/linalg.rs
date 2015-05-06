//! Test that:
//!
//! `mat[r, :][c] == mat[r, c]`
//!
//! for any valid `r`, `c`

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

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

        test_eq!(v.row(row)[col], v[(row, col)])
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

        test_eq!(v.row_mut(row)[col], v[(row, col)])
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

    let m = setup::mat((srow + nrows, scol + ncols));
    let v = m.slice((srow.., scol..));

    test_eq!(v.row(row)[col], v[(row, col)])
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

    let mut m = setup::mat((srow + nrows, scol + ncols));
    let mut v = m.slice_mut((srow.., scol..));

    test_eq!(v.row_mut(row)[col], v[(row, col)])
}
