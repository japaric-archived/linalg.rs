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

mod col {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `at_mut(_)` is correct for `ColVec`
    #[quickcheck]
    fn owned(size: usize, idx: usize) -> TestResult {
        enforce! {
            idx < size,
        }

        test!({
            let mut c = setup::col(size);
            let &mut e = try!(c.at_mut(idx));

            e == idx
        })
    }

    // Test that `at_mut(_)` is correct for `MutCol`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let mut c = try!(m.col_mut(col));
            let &mut e = try!(c.at_mut(row));

            e == (row, col)
        })
    }

    // Test that `at_mut(_)` is correct for `strided::MutCol`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let mut c = try!(m.col_mut(col));
            let &mut e = try!(c.at_mut(row));

            e == (col, row)
        })
    }
}

mod diag {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `at_mut(_)` is correct for `MutDiag`
    #[quickcheck]
    fn strided_mut(size: (usize, usize), (diag, idx): (isize, usize)) -> TestResult {
        validate_diag_index!(diag, size, idx);

        test!({
            let mut m = setup::mat(size);
            let mut c = try!(m.diag_mut(diag));
            let &mut e = try!(c.at_mut(idx));

            e == if diag > 0 {
                (idx, idx + diag as usize)
            } else {
                (idx - diag as usize, idx)
            }
        })
    }
}

mod row {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `at_mut(_)` is correct for `RowVec`
    #[quickcheck]
    fn owned(size: usize, idx: usize) -> TestResult {
        enforce! {
            idx < size,
        }

        test!({
            let mut r = setup::row(size);
            let &mut e = try!(r.at_mut(idx));

            e == idx
        })
    }

    // Test that `at_mut(_)` is correct for `MutRow`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let mut r = try!(m.row_mut(row));
            let &mut e = try!(r.at_mut(col));

            e == (col, row)
        })
    }

    // Test that `at_mut(_)` is correct for `strided::MutRow`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let mut r = try!(m.row_mut(row));
            let &mut e = try!(r.at_mut(col));

            e == (row, col)
        })
    }
}

mod trans {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `at_mut(_)` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let &mut e = try!(m.at_mut((row, col)));

            e == (col, row)
        })
    }

    // Test that `at_mut(_)` is correct for `Trans<MutView>`
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
            let mut v = try!(m.slice_from_mut(start)).t();
            let &mut e = try!(v.at_mut((row, col)));
            let (start_row, start_col) = start;

            e == (start_row + col, start_col + row)
        })
    }
}

// Test that `at_mut(_)` is correct for `Mat`
#[quickcheck]
fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
    enforce! {
        row < nrows,
        col < ncols,
    }

    test!({
        let mut m = setup::mat((nrows, ncols));
        let &mut e = try!(m.at_mut((row, col)));

        e == (row, col)
    })
}

// Test that `at_mut(_)` is correct for `MutView`
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
        let mut v = try!(m.slice_from_mut(start));
        let &mut e = try!(v.at_mut((row, col)));
        let (start_row, start_col) = start;

        e == (start_row + row, start_col + col)
    })
}
