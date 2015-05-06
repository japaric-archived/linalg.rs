//! Test that:
//!
//! - `mat.rows[_mut]().count() == mat.nrows()`
//! - `mat.rows[_mut]().enumerate().all(|(r, row)| row[c] == mat[r, c])`
//!
//! for any valid `c`

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

mod transposed {
    use cast::From;
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn submat((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), col: u32) -> TestResult {
        enforce! {
            col < ncols,
        }

        let m = ::setup::mat((srow + ncols, scol + nrows));
        let v = m.slice((srow.., scol..)).t();

        let mut row = 0;
        let mut rows = v.rows();

        test_eq! {
            rows.size_hint(),
            (usize::from(nrows - row), Some(usize::from(nrows - row)))
        };
        while let Some(r) = rows.next() {
            test_eq!(&r[col], &(srow + col, scol + row));

            row += 1;

            test_eq! {
                rows.size_hint(),
                (usize::from(nrows - row), Some(usize::from(nrows - row)))
            };
        }

        test_eq!(row, nrows)
    }

    #[quickcheck]
    fn submat_mut((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), col: u32) -> TestResult {
        enforce! {
            col < ncols,
        }

        let mut m = ::setup::mat((srow + ncols, scol + nrows));
        let mut v = m.slice_mut((srow.., scol..)).t();

        let mut row = 0;
        let mut rows = v.rows_mut();

        test_eq! {
            rows.size_hint(),
            (usize::from(nrows - row), Some(usize::from(nrows - row)))
        };
        while let Some(mut r) = rows.next() {
            test_eq!(&mut r[col], &mut (srow + col, scol + row));

            row += 1;

            test_eq! {
                rows.size_hint(),
                (usize::from(nrows - row), Some(usize::from(nrows - row)))
            };
        }

        test_eq!(row, nrows)
    }
}

#[quickcheck]
fn submat((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), col: u32) -> TestResult {
    enforce! {
        col < ncols,
    }

    let m = setup::mat((srow + nrows, scol + ncols));
    let v = m.slice((srow.., scol..));

    let mut row = 0;
    let mut rows = v.rows();

    test_eq! {
        rows.size_hint(),
        (usize::from(nrows - row), Some(usize::from(nrows - row)))
    };
    while let Some(r) = rows.next() {
        test_eq!(&r[col], &(srow + row, scol + col));

        row += 1;

        test_eq! {
            rows.size_hint(),
            (usize::from(nrows - row), Some(usize::from(nrows - row)))
        };
    }

    test_eq!(row, nrows)
}

#[quickcheck]
fn submat_mut((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), col: u32) -> TestResult {
    enforce! {
        col < ncols,
    }

    let mut m = setup::mat((srow + nrows, scol + ncols));
    let mut v = m.slice_mut((srow.., scol..));

    let mut row = 0;
    let mut rows = v.rows_mut();

    test_eq! {
        rows.size_hint(),
        (usize::from(nrows - row), Some(usize::from(nrows - row)))
    };
    while let Some(mut r) = rows.next() {
        test_eq!(&mut r[col], &mut (srow + row, scol + col));

        row += 1;

        test_eq! {
            rows.size_hint(),
            (usize::from(nrows - row), Some(usize::from(nrows - row)))
        };
    }

    test_eq!(row, nrows)
}
