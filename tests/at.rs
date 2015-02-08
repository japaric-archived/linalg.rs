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

    // Test that `at(_)` is correct for `ColVec`
    #[quickcheck]
    fn owned(size: usize, idx: usize) -> TestResult {
        enforce! {
            idx < size,
        }

        test!({
            let c = setup::col(size);
            let &e = try!(c.at(idx));

            e == idx
        })
    }

    // Test that `at(_)` is correct for `Col`
    #[quickcheck]
    fn slice((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let m = setup::mat((nrows, ncols));
            let c = try!(m.col(col));
            let &e = try!(c.at(row));

            e == (row, col)
        })
    }

    // Test that `at(_)` is correct for `MutCol`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let c = try!(m.col_mut(col));
            let &e = try!(c.at(row));

            e == (row, col)
        })
    }

    // Test that `at(_)` is correct for `strided::Col`
    #[quickcheck]
    fn strided((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let c = try!(m.col(col));
            let &e = try!(c.at(row));

            e == (col, row)
        })
    }

    // Test that `at(_)` is correct for `strided::MutCol`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let c = try!(m.col_mut(col));
            let &e = try!(c.at(row));

            e == (col, row)
        })
    }
}

mod diag {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `at(_)` is correct for `Diag`
    #[quickcheck]
    fn strided(size: (usize, usize), (diag, idx): (isize, usize)) -> TestResult {
        validate_diag_index!(diag, size, idx);

        test!({
            let m = setup::mat(size);
            let c = try!(m.diag(diag));
            let &e = try!(c.at(idx));

            e == if diag > 0 {
                (idx, idx + diag as usize)
            } else {
                (idx - diag as usize, idx)
            }
        })
    }

    // Test that `at(_)` is correct for `MutDiag`
    #[quickcheck]
    fn strided_mut(size: (usize, usize), (diag, idx): (isize, usize)) -> TestResult {
        validate_diag_index!(diag, size, idx);

        test!({
            let mut m = setup::mat(size);
            let c = try!(m.diag_mut(diag));
            let &e = try!(c.at(idx));

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

    // Test that `at(_)` is correct for `RowVec`
    #[quickcheck]
    fn owned(size: usize, idx: usize) -> TestResult {
        enforce! {
            idx < size,
        }

        test!({
            let r = setup::row(size);
            let &e = try!(r.at(idx));

            e == idx
        })
    }

    // Test that `at(_)` is correct for `Row`
    #[quickcheck]
    fn slice((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
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

    // Test that `at(_)` is correct for `MutRow`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let r = try!(m.row_mut(row));
            let &e = try!(r.at(col));

            e == (col, row)
        })
    }

    // Test that `at(_)` is correct for `strided::Row`
    #[quickcheck]
    fn strided((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
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

    // Test that `at(_)` is correct for `strided::MutRow`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let r = try!(m.row_mut(row));
            let &e = try!(r.at(col));

            e == (row, col)
        })
    }
}

mod trans {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `at(_)` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let &e = try!(m.at((row, col)));

            e == (col, row)
        })
    }

    // Test that `at(_)` is correct for `Trans<View>`
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
            let v = try!(m.slice_from(start)).t();
            let &e = try!(v.at((row, col)));
            let (start_row, start_col) = start;

            e == (start_row + col, start_col + row)
        })
    }

    // Test that `at(_)` is correct for `Trans<MutView>`
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
            let v = try!(m.slice_from_mut(start)).t();
            let &e = try!(v.at((row, col)));
            let (start_row, start_col) = start;

            e == (start_row + col, start_col + row)
        })
    }
}

// Test that `at(_)` is correct for `Mat`
#[quickcheck]
fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
    enforce! {
        row < nrows,
        col < ncols,
    }

    test!({
        let m = setup::mat((nrows, ncols));
        let &e = try!(m.at((row, col)));

        e == (row, col)
    })
}

// Test that `at(_)` is correct for `View`
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
        let v = try!(m.slice_from(start));
        let &e = try!(v.at((row, col)));
        let (start_row, start_col) = start;

        e == (start_row + row, start_col + col)
    })
}

// Test that `at(_)` is correct for `MutView`
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
        let v = try!(m.slice_from_mut(start));
        let &e = try!(v.at((row, col)));
        let (start_row, start_col) = start;

        e == (start_row + row, start_col + col)
    })
}
