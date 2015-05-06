//! Test that:
//!
//! - `mat.cols[_mut]().count() == mat.ncols()`
//! - `mat.cols[_mut]().enumerate().all(|(c, col)| col[r] == mat[r, c])`
//!
//! for any valid `r`

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
    fn submat((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), row: u32) -> TestResult {
        enforce! {
            row < nrows,
        }

        let m = ::setup::mat((srow + ncols, scol + nrows));
        let v = m.slice((srow.., scol..)).t();

        let mut col = 0;
        let mut cols = v.cols();

        test_eq! {
            cols.size_hint(),
            (usize::from(ncols - col), Some(usize::from(ncols - col)))
        };
        while let Some(c) = cols.next() {
            test_eq!(&c[row], &(srow + col, scol + row));

            col += 1;

            test_eq! {
                cols.size_hint(),
                (usize::from(ncols - col), Some(usize::from(ncols - col)))
            };
        }

        test_eq!(col, ncols)
    }

    #[quickcheck]
    fn submat_mut((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), row: u32) -> TestResult {
        enforce! {
            row < nrows,
        }

        let mut m = ::setup::mat((srow + ncols, scol + nrows));
        let mut v = m.slice_mut((srow.., scol..)).t();

        let mut col = 0;
        let mut cols = v.cols_mut();

        test_eq! {
            cols.size_hint(),
            (usize::from(ncols - col), Some(usize::from(ncols - col)))
        };
        while let Some(mut c) = cols.next() {
            test_eq!(&mut c[row], &mut (srow + col, scol + row));

            col += 1;

            test_eq! {
                cols.size_hint(),
                (usize::from(ncols - col), Some(usize::from(ncols - col)))
            };
        }

        test_eq!(col, ncols)
    }
}

#[quickcheck]
fn submat((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), row: u32) -> TestResult {
    enforce! {
        row < nrows,
    }

    let m = setup::mat((srow + nrows, scol + ncols));
    let v = m.slice((srow.., scol..));

    let mut col = 0;
    let mut cols = v.cols();

    test_eq! {
        cols.size_hint(),
        (usize::from(ncols - col), Some(usize::from(ncols - col)))
    };
    while let Some(c) = cols.next() {
        test_eq!(&c[row], &(srow + row, scol + col));

        col += 1;

        test_eq! {
            cols.size_hint(),
            (usize::from(ncols - col), Some(usize::from(ncols - col)))
        };
    }

    test_eq!(col, ncols)
}

#[quickcheck]
fn submat_mut((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), row: u32) -> TestResult {
    enforce! {
        row < nrows,
    }

    let mut m = setup::mat((srow + nrows, scol + ncols));
    let mut v = m.slice_mut((srow.., scol..));

    let mut col = 0;
    let mut cols = v.cols_mut();

    test_eq! {
        cols.size_hint(),
        (usize::from(ncols - col), Some(usize::from(ncols - col)))
    };
    while let Some(mut c) = cols.next() {
        test_eq!(&mut c[row], &mut (srow + row, scol + col));

        col += 1;

        test_eq! {
            cols.size_hint(),
            (usize::from(ncols - col), Some(usize::from(ncols - col)))
        };
    }

    test_eq!(col, ncols)
}
