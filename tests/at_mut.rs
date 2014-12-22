#![feature(globs, macro_rules, phase)]

extern crate linalg;
extern crate quickcheck;
#[phase(plugin)]
extern crate quickcheck_macros;

use linalg::prelude::*;
use quickcheck::TestResult;

mod setup;

mod col {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `at_mut(_)` is correct for `ColVec`
    #[quickcheck]
    fn owned(size: uint, idx: uint) -> TestResult {
        enforce! {
            idx < size,
        }

        test!({
            let mut c = setup::col(size);
            let &e = try!(c.at_mut(idx));

            e == idx
        })
    }

    // Test that `at_mut(_)` is correct for `MutCol`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (uint, uint), (row, col): (uint, uint)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let mut c = try!(m.col_mut(col));
            let &e = try!(c.at_mut(row));

            e == (row, col)
        })
    }

    // Test that `at_mut(_)` is correct for `strided::MutCol`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (uint, uint), (row, col): (uint, uint)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let mut c = try!(m.col_mut(col));
            let &e = try!(c.at_mut(row));

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
    fn strided_mut(size: (uint, uint), (diag, idx): (int, uint)) -> TestResult {
        validate_diag_index!(diag, size, idx);

        test!({
            let mut m = setup::mat(size);
            let mut c = try!(m.diag_mut(diag));
            let &e = try!(c.at_mut(idx));

            e == if diag > 0 {
                (idx, idx + diag as uint)
            } else {
                (idx - diag as uint, idx)
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
    fn owned(size: uint, idx: uint) -> TestResult {
        enforce! {
            idx < size,
        }

        test!({
            let mut r = setup::row(size);
            let &e = try!(r.at_mut(idx));

            e == idx
        })
    }

    // Test that `at_mut(_)` is correct for `MutRow`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (uint, uint), (row, col): (uint, uint)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let mut r = try!(m.row_mut(row));
            let &e = try!(r.at_mut(col));

            e == (col, row)
        })
    }

    // Test that `at_mut(_)` is correct for `strided::MutRow`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (uint, uint), (row, col): (uint, uint)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let mut r = try!(m.row_mut(row));
            let &e = try!(r.at_mut(col));

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
    fn mat((nrows, ncols): (uint, uint), (row, col): (uint, uint)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let &e = try!(m.at_mut((row, col)));

            e == (col, row)
        })
    }

    // Test that `at_mut(_)` is correct for `Trans<MutView>`
    #[quickcheck]
    fn view_mut(
        start: (uint, uint),
        (nrows, ncols): (uint, uint),
        (row, col): (uint, uint),
    ) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let mut m = setup::mat(size);
            let mut v = try!(m.slice_from_mut(start)).t();
            let &e = try!(v.at_mut((row, col)));
            let (start_row, start_col) = start;

            e == (start_row + col, start_col + row)
        })
    }
}

// Test that `at_mut(_)` is correct for `Mat`
#[quickcheck]
fn mat((nrows, ncols): (uint, uint), (row, col): (uint, uint)) -> TestResult {
    enforce! {
        row < nrows,
        col < ncols,
    }

    test!({
        let mut m = setup::mat((nrows, ncols));
        let &e = try!(m.at_mut((row, col)));

        e == (row, col)
    })
}

// Test that `at_mut(_)` is correct for `MutView`
#[quickcheck]
fn view_mut(
    start: (uint, uint),
    (nrows, ncols): (uint, uint),
    (row, col): (uint, uint),
) -> TestResult {
    enforce! {
        row < nrows,
        col < ncols,
    }

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let mut m = setup::mat(size);
        let mut v = try!(m.slice_from_mut(start));
        let &e = try!(v.at_mut((row, col)));
        let (start_row, start_col) = start;

        e == (start_row + row, start_col + col)
    })
}
