//! Given:
//!
//! A matrix where each element is equal to its index
//!
//! Test that:
//!
//! `&mut mat[r, c] == &mut (r, c)` for any valid `r`, `c`

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate cast;
extern crate linalg;
extern crate quickcheck;
extern crate rand;

use cast::From;

use linalg::prelude::*;
use quickcheck::TestResult;

#[macro_use]
mod setup;

mod col {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn mut_contiguous((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let mut m = ::setup::mat((nrows, ncols));
        let mut c = m.col_mut(col);

        test_eq!(&mut c[row], &mut (row, col))
    }

    #[quickcheck]
    fn mut_strided((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let mut m = ::setup::mat((ncols, nrows)).t();
        let mut c = m.col_mut(col);

        test_eq!(&mut c[row], &mut (col, row))
    }

    #[quickcheck]
    fn vec(len: u32, mut i: u32) -> TestResult {
        enforce! {
            i < len,
        }

        let mut c = ::setup::col(len);

        test_eq!(&mut c[i], &mut i)
    }
}

mod row {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn mut_contiguous((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let mut m = ::setup::mat((ncols, nrows)).t();
        let mut r = m.row_mut(row);

        test_eq!(&mut r[col], &mut (col, row))
    }

    #[quickcheck]
    fn mut_strided((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let mut m = ::setup::mat((nrows, ncols));
        let mut r = m.row_mut(row);

        test_eq!(&mut r[col], &mut (row, col))
    }

    #[quickcheck]
    fn vec(len: u32, mut i: u32) -> TestResult {
        enforce! {
            i < len,
        }

        let mut c = ::setup::row(len);

        test_eq!(&mut c[i], &mut i)
    }
}

mod transposed {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn mat((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let mut m = ::setup::mat((ncols, nrows)).t();

        test_eq!(&mut m[(row, col)], &mut (col, row))
    }

    #[quickcheck]
    fn submat_mut(
        (srow, scol): (u32, u32),
        (nrows, ncols): (u32, u32),
        (row, col): (u32, u32),
    ) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let mut m = ::setup::mat((srow + ncols, scol + nrows));
        let mut v = m.slice_mut((srow.., scol..)).t();

        test_eq!(&mut v[(row, col)], &mut (srow + col, scol + row))
    }
}

#[quickcheck]
fn diag_mut(size: (u32, u32), (diag, i): (i32, u32)) -> TestResult {
    validate_diag_index!(size, diag, i);

    let mut m = ::setup::mat(size);
    let mut d = m.diag_mut(diag);

    test_eq!(&mut d[i], &mut if diag > 0 {
        (i, i + u32::from(diag).unwrap())
    } else {
        (i + u32::from(-diag).unwrap(), i)
    })
}

#[quickcheck]
fn mat((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
    enforce! {
        col < ncols,
        row < nrows,
    }

    let mut m = setup::mat((nrows, ncols));

    test_eq!(&mut m[(row, col)], &mut (row, col))
}

#[quickcheck]
fn submat_mut(
    (srow, scol): (u32, u32),
    (nrows, ncols): (u32, u32),
    (row, col): (u32, u32),
) -> TestResult {
    enforce! {
        col < ncols,
        row < nrows,
    }

    let mut m = setup::mat((srow + nrows, scol + ncols));
    let mut v = m.slice_mut((srow.., scol..));

    test_eq!(&mut v[(row, col)], &mut (srow + row, scol + col))
}
