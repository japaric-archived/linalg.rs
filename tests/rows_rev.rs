#![feature(plugin)]
#![feature(rand)]

extern crate linalg;
extern crate quickcheck;
#[plugin]
extern crate quickcheck_macros;

use linalg::prelude::*;
use quickcheck::TestResult;

#[macro_use]
mod setup;

mod trans {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `rows().rev()` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat((nrows, ncols): (usize, usize), col: usize) -> TestResult {
        enforce! {
            col < ncols,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let n = m.nrows();

            m.rows().rev().enumerate().all(|(i, r)| {
                let i = n - i - 1;

                r.at(col).unwrap() == &(col, i)
            })
        })
    }

    // Test that `rows().rev()` is correct for `Trans<View>`
    #[quickcheck]
    fn view(
        start: (usize, usize),
        (nrows, ncols): (usize, usize),
        col: usize,
    ) -> TestResult {
        enforce! {
            col < ncols,
        }

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let m = setup::mat(size);
            let v = try!(m.slice_from(start)).t();
            let (start_row, start_col) = start;
            let n = v.nrows();

            v.rows().rev().enumerate().all(|(i, r)| {
                let i = n - i - 1;

                r.at(col).unwrap() == &(start_row + col, start_col + i)
            })
        })
    }

    // Test that `rows().rev()` is correct for `Trans<MutView>`
    #[quickcheck]
    fn view_mut(
        start: (usize, usize),
        (nrows, ncols): (usize, usize),
        col: usize,
    ) -> TestResult {
        enforce! {
            col < ncols,
        }

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let mut m = setup::mat(size);
            let v = try!(m.slice_from_mut(start)).t();
            let (start_row, start_col) = start;
            let n = v.nrows();

            v.rows().rev().enumerate().all(|(i, r)| {
                let i = n - i - 1;

                r.at(col).unwrap() == &(start_row + col, start_col + i)
            })
        })
    }
}

// Test that `rows().rev()` is correct for `Mat`
#[quickcheck]
fn mat((nrows, ncols): (usize, usize), col: usize) -> TestResult {
    enforce! {
        col < ncols,
    }

    test!({
        let m = setup::mat((nrows, ncols));
        let n = m.nrows();

        m.rows().rev().enumerate().all(|(i, r)| {
            let i = n - i - 1;

            r.at(col).unwrap() == &(i, col)
        })
    })
}

// Test that `rows().rev()` is correct for `View`
#[quickcheck]
fn view(
    start: (usize, usize),
    (nrows, ncols): (usize, usize),
    col: usize,
) -> TestResult {
    enforce! {
        col < ncols,
    }

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let m = setup::mat(size);
        let v = try!(m.slice_from(start));
        let (start_row, start_col) = start;
        let n = v.nrows();

        v.rows().rev().enumerate().all(|(i, r)| {
            let i = n - i - 1;

            r.at(col).unwrap() == &(start_row + i, start_col + col)
        })
    })
}

// Test that `rows().rev()` is correct for `MutView`
#[quickcheck]
fn view_mut(
    start: (usize, usize),
    (nrows, ncols): (usize, usize),
    col: usize,
) -> TestResult {
    enforce! {
        col < ncols,
    }

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let mut m = setup::mat(size);
        let v = try!(m.slice_from_mut(start));
        let (start_row, start_col) = start;
        let n = v.nrows();

        v.rows().rev().enumerate().all(|(i, r)| {
            let i = n - i - 1;

            r.at(col).unwrap() == &(start_row + i, start_col + col)
        })
    })
}
