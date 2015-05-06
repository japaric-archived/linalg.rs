//! Given:
//!
//! A matrix where each element is equal to its index
//!
//! Test that:
//!
//! `&mat[r, c] == &(r, c)` for any valid `r`, `c`

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
    fn contiguous((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let m = ::setup::mat((nrows, ncols));
        let c = m.col(col);

        test_eq!(&c[row], &(row, col))
    }

    #[quickcheck]
    fn strided((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let m = ::setup::mat((ncols, nrows)).t();
        let c = m.col(col);

        test_eq!(&c[row], &(col, row))
    }

    #[quickcheck]
    fn mut_contiguous((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let mut m = ::setup::mat((nrows, ncols));
        let c = m.col_mut(col);

        test_eq!(&c[row], &(row, col))
    }

    #[quickcheck]
    fn mut_strided((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let mut m = ::setup::mat((ncols, nrows)).t();
        let c = m.col_mut(col);

        test_eq!(&c[row], &(col, row))
    }

    #[quickcheck]
    fn vec(len: u32, i: u32) -> TestResult {
        enforce! {
            i < len,
        }

        let c = ::setup::col(len);

        test_eq!(&c[i], &i)
    }
}

mod row {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn contiguous((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let m = ::setup::mat((ncols, nrows)).t();
        let r = m.row(row);

        test_eq!(&r[col], &(col, row))
    }

    #[quickcheck]
    fn strided((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let m = ::setup::mat((nrows, ncols));
        let r = m.row(row);

        test_eq!(&r[col], &(row, col))
    }

    #[quickcheck]
    fn mut_contiguous((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let mut m = ::setup::mat((ncols, nrows)).t();
        let r = m.row_mut(row);

        test_eq!(&r[col], &(col, row))
    }

    #[quickcheck]
    fn mut_strided((nrows, ncols): (u32, u32), (row, col): (u32, u32)) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let mut m = ::setup::mat((nrows, ncols));
        let r = m.row_mut(row);

        test_eq!(&r[col], &(row, col))
    }

    #[quickcheck]
    fn vec(len: u32, i: u32) -> TestResult {
        enforce! {
            i < len,
        }

        let c = ::setup::row(len);

        test_eq!(&c[i], &i)
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

        let m = ::setup::mat((ncols, nrows)).t();

        test_eq!(&m[(row, col)], &(col, row))
    }

    #[quickcheck]
    fn submat(
        (srow, scol): (u32, u32),
        (nrows, ncols): (u32, u32),
        (row, col): (u32, u32),
    ) -> TestResult {
        enforce! {
            col < ncols,
            row < nrows,
        }

        let m = ::setup::mat((srow + ncols, scol + nrows));
        let v = m.slice((srow.., scol..)).t();

        test_eq!(&v[(row, col)], &(srow + col, scol + row))
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
        let v = m.slice_mut((srow.., scol..)).t();

        test_eq!(&v[(row, col)], &(srow + col, scol + row))
    }
}

#[quickcheck]
fn diag(size: (u32, u32), (diag, i): (i32, u32)) -> TestResult {
    validate_diag_index!(size, diag, i);

    let m = ::setup::mat(size);
    let d = m.diag(diag);

    test_eq!(&d[i], & if diag > 0 {
        (i, i + u32::from(diag).unwrap())
    } else {
        (i + u32::from(-diag).unwrap(), i)
    })
}

#[quickcheck]
fn diag_mut(size: (u32, u32), (diag, i): (i32, u32)) -> TestResult {
    validate_diag_index!(size, diag, i);

    let mut m = ::setup::mat(size);
    let d = m.diag_mut(diag);

    test_eq!(&d[i], & if diag > 0 {
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

    let m = setup::mat((nrows, ncols));

    test_eq!(&m[(row, col)], &(row, col))
}

#[quickcheck]
fn submat(
    (srow, scol): (u32, u32),
    (nrows, ncols): (u32, u32),
    (row, col): (u32, u32),
) -> TestResult {
    enforce! {
        col < ncols,
        row < nrows,
    }

    let m = setup::mat((srow + nrows, scol + ncols));
    let v = m.slice((srow.., scol..));

    test_eq!(&v[(row, col)], &(srow + row, scol + col))
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
    let v = m.slice_mut((srow.., scol..));

    test_eq!(&v[(row, col)], &(srow + row, scol + col))
}
