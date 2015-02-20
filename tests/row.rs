#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate linalg;
extern crate quickcheck;
extern crate rand;

use linalg::prelude::*;
use quickcheck::TestResult;

#[macro_use]
mod setup;

mod trans {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `row(_)` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let r = try!(m.row(row));
            let &e = try!(r.at(col));

            e == (col, row)
        })
    }

    // Test that `row(_)` is correct for `Trans<View>`
    #[quickcheck]
    fn view(
        start: (usize, usize),
        (nrows, ncols): (usize, usize),
        (row, col): (usize, usize),
    ) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let m = setup::mat(size);
            let v = try!(m.slice(start..)).t();
            let r = try!(v.row(row));
            let &e = try!(r.at(col));
            let (start_row, start_col) = start;

            e == (start_row + col, start_col + row)
        })
    }

    // Test that `row(_)` is correct for `Trans<MutView>`
    #[quickcheck]
    fn view_mut(
        start: (usize, usize),
        (nrows, ncols): (usize, usize),
        (row, col): (usize, usize),
    ) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let mut m = setup::mat(size);
            let v = try!(m.slice_mut(start..)).t();
            let r = try!(v.row(row));
            let &e = try!(r.at(col));
            let (start_row, start_col) = start;

            e == (start_row + col, start_col + row)
        })
    }
}

// Test that `row(_)` is correct for `Mat`
#[quickcheck]
fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
    enforce! {
        row < nrows,
        col < ncols,
    }

    test!({
        let m = setup::mat((nrows, ncols));
        let r = try!(m.row(row));
        let &e = try!(r.at(col));

        e == (row, col)
    })
}

// Test that `row(_)` is correct for `View`
#[quickcheck]
fn view(
    start: (usize, usize),
    (nrows, ncols): (usize, usize),
    (row, col): (usize, usize),
) -> TestResult {
    enforce! {
        row < nrows,
        col < ncols,
    }

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let m = setup::mat(size);
        let v = try!(m.slice(start..));
        let r = try!(v.row(row));
        let &e = try!(r.at(col));
        let (start_row, start_col) = start;

        e == (start_row + row, start_col + col)
    })
}

// Test that `row(_)` is correct for `MutView`
#[quickcheck]
fn view_mut(
    start: (usize, usize),
    (nrows, ncols): (usize, usize),
    (row, col): (usize, usize),
) -> TestResult {
    enforce! {
        row < nrows,
        col < ncols,
    }

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let mut m = setup::mat(size);
        let v = try!(m.slice_mut(start..));
        let r = try!(v.row(row));
        let &e = try!(r.at(col));
        let (start_row, start_col) = start;

        e == (start_row + row, start_col + col)
    })
}
