#![feature(plugin)]

extern crate linalg;
extern crate quickcheck;
#[plugin]
extern crate quickcheck_macros;
extern crate rand;

use linalg::prelude::*;
use quickcheck::TestResult;

#[macro_use]
mod setup;

mod trans {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `cols()` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat((nrows, ncols): (usize, usize), row: usize) -> TestResult {
        enforce! {
            row < nrows,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();

            m.cols().enumerate().all(|(i, c)| c.at(row).unwrap() == &(i, row))
        })
    }

    // Test that `cols()` is correct for `Trans<View>`
    #[quickcheck]
    fn view(
        start: (usize, usize),
        (nrows, ncols): (usize, usize),
        row: usize,
    ) -> TestResult {
        enforce! {
            row < nrows,
        }

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let m = setup::mat(size);
            let v = try!(m.slice_from(start)).t();
            let (start_row, start_col) = start;

            v.cols().enumerate().all(|(i, c)| {
                c.at(row).unwrap() == &(start_row + i, start_col + row)
            })
        })
    }

    // Test that `cols()` is correct for `Trans<MutView>`
    #[quickcheck]
    fn view_mut(
        start: (usize, usize),
        (nrows, ncols): (usize, usize),
        row: usize,
    ) -> TestResult {
        enforce! {
            row < nrows,
        }

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let mut m = setup::mat(size);
            let v = try!(m.slice_from_mut(start)).t();
            let (start_row, start_col) = start;

            v.cols().enumerate().all(|(i, c)| {
                c.at(row).unwrap() == &(start_row + i, start_col + row)
            })
        })
    }
}

// Test that `cols()` is correct for `Mat`
#[quickcheck]
fn mat((nrows, ncols): (usize, usize), row: usize) -> TestResult {
    enforce! {
        row < nrows,
    }

    test!({
        let m = setup::mat((nrows, ncols));

        m.cols().enumerate().all(|(i, c)| c.at(row).unwrap() == &(row, i))
    })
}

// Test that `cols()` is correct for `View`
#[quickcheck]
fn view(
    start: (usize, usize),
    (nrows, ncols): (usize, usize),
    row: usize,
) -> TestResult {
    enforce! {
        row < nrows,
    }

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let m = setup::mat(size);
        let v = try!(m.slice_from(start));
        let (start_row, start_col) = start;

        v.cols().enumerate().all(|(i, c)| c.at(row).unwrap() == &(start_row + row, start_col + i))
    })
}

// Test that `cols()` is correct for `MutView`
#[quickcheck]
fn view_mut(
    start: (usize, usize),
    (nrows, ncols): (usize, usize),
    row: usize,
) -> TestResult {
    enforce! {
        row < nrows,
    }

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let mut m = setup::mat(size);
        let v = try!(m.slice_from_mut(start));
        let (start_row, start_col) = start;

        v.cols().enumerate().all(|(i, c)| c.at(row).unwrap() == &(start_row + row, start_col + i))
    })
}
