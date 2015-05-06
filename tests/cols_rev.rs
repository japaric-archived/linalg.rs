//! Test that:
//!
//! - `mat.cols[_mut]().rev().count() == mat.ncols()`
//! - `mat.cols[_mut]().rev().enumerate().all(|(c, col)| col[r] == mat[r, nc - c - 1])`
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

        let mut col = ncols;
        let mut cols = v.cols().rev();

        test_eq!(cols.size_hint(), (usize::from(col), Some(usize::from(col))));
        while let Some(c) = cols.next() {
            col -= 1;

            test_eq!(cols.size_hint(), (usize::from(col), Some(usize::from(col))));
            test_eq!(&c[row], &(srow + col, scol + row));
        }

        test_eq!(col, 0)
    }

    #[quickcheck]
    fn submat_mut((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), row: u32) -> TestResult {
        enforce! {
            row < nrows,
        }

        let mut m = ::setup::mat((srow + ncols, scol + nrows));
        let mut v = m.slice_mut((srow.., scol..)).t();

        let mut col = ncols;
        let mut cols = v.cols_mut().rev();

        test_eq!(cols.size_hint(), (usize::from(col), Some(usize::from(col))));
        while let Some(mut c) = cols.next() {
            col -= 1;

            test_eq!(cols.size_hint(), (usize::from(col), Some(usize::from(col))));
            test_eq!(&mut c[row], &mut (srow + col, scol + row));
        }

        test_eq!(col, 0)
    }
}

#[quickcheck]
fn submat((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), row: u32) -> TestResult {
    enforce! {
        row < nrows,
    }

    let m = setup::mat((srow + nrows, scol + ncols));
    let v = m.slice((srow.., scol..));

    let mut col = ncols;
    let mut cols = v.cols().rev();

    test_eq!(cols.size_hint(), (usize::from(col), Some(usize::from(col))));
    while let Some(c) = cols.next() {
        col -= 1;

        test_eq!(cols.size_hint(), (usize::from(col), Some(usize::from(col))));
        test_eq!(&c[row], &(srow + row, scol + col));
    }

    test_eq!(col, 0)
}

#[quickcheck]
fn submat_mut((srow, scol): (u32, u32), (nrows, ncols): (u32, u32), row: u32) -> TestResult {
    enforce! {
        row < nrows,
    }

    let mut m = setup::mat((srow + nrows, scol + ncols));
    let mut v = m.slice_mut((srow.., scol..));

    let mut col = ncols;
    let mut cols = v.cols_mut().rev();

    test_eq!(cols.size_hint(), (usize::from(col), Some(usize::from(col))));
    while let Some(mut c) = cols.next() {
        col -= 1;

        test_eq!(cols.size_hint(), (usize::from(col), Some(usize::from(col))));
        test_eq!(&mut c[row], &mut (srow + row, scol + col));
    }

    test_eq!(col, 0)
}
