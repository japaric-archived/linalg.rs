#![feature(globs, macro_rules, phase)]

extern crate linalg;
extern crate quickcheck;
#[phase(plugin)]
extern crate quickcheck_macros;

use linalg::prelude::*;
use quickcheck::TestResult;

use utils::IsWithin;

mod setup;
mod utils;

mod col {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `slice_mut(_, _)` is correct for `ColVec`
    #[quickcheck]
    fn owned(size: uint, (start, end): (uint, uint), idx: uint) -> TestResult {
        enforce! {
            end <= size,
            start <= end,
            idx < end - start,
        }

        test!({
            let mut c = setup::col(size);
            let s = try!(c.slice_mut(start, end));
            let &e = try!(s.at(idx));

            e == start + idx
        })
    }

    // Test that `slice_mut(_, _)` is correct for `MutCol`
    #[quickcheck]
    fn slice_mut(
        (nrows, ncols): (uint, uint),
        col: uint,
        (start, end): (uint, uint),
        idx: uint,
    ) -> TestResult {
        enforce! {
            col < ncols,
            end <= nrows,
            start <= end,
            idx < end - start,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let mut c = try!(m.col_mut(col));
            let s = try!(c.slice_mut(start, end));
            let &e = try!(s.at(idx));

            e == (start + idx, col)
        })
    }

    // Test that `slice_mut(_, _)` is correct for `strided::MutCol`
    #[quickcheck]
    fn strided_mut(
        (nrows, ncols): (uint, uint),
        col: uint,
        (start, end): (uint, uint),
        idx: uint,
    ) -> TestResult {
        enforce! {
            col < ncols,
            end <= nrows,
            start <= end,
            idx < end - start,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let mut c = try!(m.col_mut(col));
            let s = try!(c.slice_mut(start, end));
            let &e = try!(s.at(idx));

            e == (col, start + idx)
        })
    }
}

mod diag {
    use linalg::prelude::*;
    use quickcheck::TestResult;
    use std::cmp;

    use setup;

    // XXX Discards too many QC tests
    // Test that `slice_mut(_, _)` is correct for `MutDiag`
    #[quickcheck]
    fn strided_mut(
        (nrows, ncols): (uint, uint),
        diag: int,
        (start, end): (uint, uint),
        idx: uint,
    ) -> TestResult {
        if diag > 0 {
            let diag = diag as uint;

            enforce! {
                diag < ncols,
                end <= cmp::min(ncols - diag, nrows),
                start <= end,
                idx < end - start,
            }
        } else {
            let diag = -diag as uint;

            enforce! {
                diag < nrows,
                end <= cmp::min(nrows - diag, ncols),
                start <= end,
                idx < end - start,
            }
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let mut c = try!(m.diag_mut(diag));
            let s = try!(c.slice_mut(start, end));
            let &e = try!(s.at(idx));
            let idx = idx + start;

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

    // Test that `slice_mut(_, _)` is correct for `RowVec`
    #[quickcheck]
    fn owned(size: uint, (start, end): (uint, uint), idx: uint) -> TestResult {
        enforce! {
            end <= size,
            start <= end,
            idx < end - start,
        }

        test!({
            let mut r = setup::row(size);
            let s = try!(r.slice_mut(start, end));
            let &e = try!(s.at(idx));

            e == start + idx
        })
    }

    // Test that `slice_mut(_, _)` is correct for `MutRow`
    #[quickcheck]
    fn slice_mut(
        (nrows, ncols): (uint, uint),
        row: uint,
        (start, end): (uint, uint),
        idx: uint,
    ) -> TestResult {
        enforce! {
            row < nrows,
            end < ncols,
            start <= end,
            idx < end - start,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let mut r = try!(m.row_mut(row));
            let s = try!(r.slice_mut(start, end));
            let &e = try!(s.at(idx));

            e == (start + idx, row)
        })
    }

    // Test that `slice_mut(_, _)` is correct for `strided::MutRow`
    #[quickcheck]
    fn strided_mut(
        (nrows, ncols): (uint, uint),
        row: uint,
        (start, end): (uint, uint),
        idx: uint,
    ) -> TestResult {
        enforce! {
            row < nrows,
            end < ncols,
            start <= end,
            idx < end - start,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let mut r = try!(m.row_mut(row));
            let s = try!(r.slice_mut(start, end));
            let &e = try!(s.at(idx));

            e == (row, start + idx)
        })
    }
}

mod trans {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;
    use utils::IsWithin;

    // XXX Discards too many QC tests
    // Test that `slice_mut(_, _)` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat(
        (nrows, ncols): (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        enforce! {
            end.is_within((nrows, ncols)),
            start.is_within(end),
            row < end.0 - start.0,
            col < end.1 - start.1,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let v = try!(m.slice_mut(start, end));
            let &e = try!(v.at((row, col)));
            let (start_row, start_col) = start;

            e == (start_col + col, start_row + row)
        })
    }

    // XXX Discards too many QC tests
    // Test that `slice_mut(_, _)` is correct for `Trans<MutView>`
    #[quickcheck]
    fn view_mut(
        start: (uint, uint),
        (nrows, ncols): (uint, uint),
        (inner_start, inner_end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        enforce! {
            inner_end.is_within((nrows, ncols)),
            inner_start.is_within(inner_end),
            row < inner_end.0 - inner_start.0,
            col < inner_end.1 - inner_start.1,
        }

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let mut m = setup::mat(size);
            let mut v = try!(m.slice_from_mut(start)).t();
            let vv = try!(v.slice_mut(inner_start, inner_end));
            let &e = try!(vv.at((row, col)));
            let (outer_start_row, outer_start_col) = start;
            let (inner_start_row, inner_start_col) = inner_start;
            let start_col = outer_start_row + inner_start_col;
            let start_row = outer_start_col + inner_start_row;

            e == (start_col + col, start_row + row)
        })
    }
}

// XXX Discards too many QC tests
// Test that `slice_mut(_, _)` is correct for `Mat`
#[quickcheck]
fn mat(
    size: (uint, uint),
    (start, end): ((uint, uint), (uint, uint)),
    (row, col): (uint, uint),
) -> TestResult {
    enforce! {
        end.is_within(size),
        start.is_within(end),
        row < end.0 - start.0,
        col < end.1 - start.1,
    }

    test!({
        let mut m = setup::mat(size);
        let v = try!(m.slice_mut(start, end));
        let &e = try!(v.at((row, col)));
        let (start_row, start_col) = start;

        e == (start_row + row, start_col + col)
    })
}

// XXX Discards too many QC tests
// Test that `slice_mut(_, _)` is correct for `MutView`
#[quickcheck]
fn view_mut(
    start: (uint, uint),
    (nrows, ncols): (uint, uint),
    (inner_start, inner_end): ((uint, uint), (uint, uint)),
    (row, col): (uint, uint),
) -> TestResult {
    enforce! {
        inner_end.is_within((nrows, ncols)),
        inner_start.is_within(inner_end),
        row < inner_end.0 - inner_start.0,
        col < inner_end.1 - inner_start.1,
    }

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let mut m = setup::mat(size);
        let mut v = try!(m.slice_from_mut(start));
        let vv = try!(v.slice_mut(inner_start, inner_end));
        let &e = try!(vv.at((row, col)));
        let (outer_start_row, outer_start_col) = start;
        let (inner_start_row, inner_start_col) = inner_start;
        let start_row = outer_start_row + inner_start_row;
        let start_col = outer_start_col + inner_start_col;

        e == (start_row + row, start_col + col)
    })
}
