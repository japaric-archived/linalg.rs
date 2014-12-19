#![feature(globs, macro_rules, phase)]

extern crate linalg;
extern crate quickcheck;
#[phase(plugin)]
extern crate quickcheck_macros;

use linalg::prelude::*;
use quickcheck::TestResult;

mod setup;

mod trans {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `cols().rev()` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce! {
            row < nrows,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let n = m.ncols();

            m.cols().rev().enumerate().all(|(i, c)| {
                let i = n - i - 1;

                c.at(row).unwrap() == &(i, row)
            })
        })
    }

    // Test that `cols().rev()` is correct for `Trans<View>`
    #[quickcheck]
    fn view(
        start: (uint, uint),
        (nrows, ncols): (uint, uint),
        row: uint,
    ) -> TestResult {
        enforce! {
            row < nrows,
        }

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let m = setup::mat(size);
            let v = try!(m.slice_from(start)).t();
            let (start_row, start_col) = start;
            let n = v.ncols();

            v.cols().rev().enumerate().all(|(i, c)| {
                let i = n - i - 1;

                c.at(row).unwrap() == &(start_row + i, start_col + row)
            })
        })
    }

    // Test that `cols().rev()` is correct for `Trans<MutView>`
    #[quickcheck]
    fn view_mut(
        start: (uint, uint),
        (nrows, ncols): (uint, uint),
        row: uint,
    ) -> TestResult {
        enforce! {
            row < nrows,
        }

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let mut m = setup::mat(size);
            let v = try!(m.slice_from_mut(start)).t();
            let (start_row, start_col) = start;
            let n = v.ncols();

            v.cols().rev().enumerate().all(|(i, c)| {
                let i = n - i - 1;

                c.at(row).unwrap() == &(start_row + i, start_col + row)
            })
        })
    }
}

// Test that `cols().rev()` is correct for `Mat`
#[quickcheck]
fn mat((nrows, ncols): (uint, uint), row: uint) -> TestResult {
    enforce! {
        row < nrows,
    }

    test!({
        let m = setup::mat((nrows, ncols));
        let n = m.ncols();

        m.cols().rev().enumerate().all(|(i, c)| {
            let i = n - i - 1;

            c.at(row).unwrap() == &(row, i)
        })
    })
}

// Test that `cols().rev()` is correct for `View`
#[quickcheck]
fn view(
    start: (uint, uint),
    (nrows, ncols): (uint, uint),
    row: uint,
) -> TestResult {
    enforce! {
        row < nrows,
    }

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let m = setup::mat(size);
        let v = try!(m.slice_from(start));
        let (start_row, start_col) = start;
        let n = v.ncols();

        v.cols().rev().enumerate().all(|(i, c)| {
            let i = n - i - 1;

            c.at(row).unwrap() == &(start_row + row, start_col + i)
        })
    })
}

// Test that `cols().rev()` is correct for `MutView`
#[quickcheck]
fn view_mut(
    start: (uint, uint),
    (nrows, ncols): (uint, uint),
    row: uint,
) -> TestResult {
    enforce! {
        row < nrows,
    }

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let mut m = setup::mat(size);
        let v = try!(m.slice_from_mut(start));
        let (start_row, start_col) = start;
        let n = v.ncols();

        v.cols().rev().enumerate().all(|(i, c)| {
            let i = n - i - 1;

            c.at(row).unwrap() == &(start_row + row, start_col + i)
        })
    })
}
