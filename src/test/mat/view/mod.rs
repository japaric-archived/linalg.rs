use quickcheck::TestResult;

use mat;
use mat::traits::MatrixView;

mod col;
mod diag;
mod row;

// Index
// FIXME rust-lang/rust#15734 Replace `index` method with `[]` operator
#[quickcheck]
fn index(shape@(nrows, ncols): (uint, uint),
         (start, end): ((uint, uint), (uint, uint)),
         index@(row, col): (uint, uint))
         -> TestResult {
    let (start_row, start_col) = start;
    let (end_row, end_col) = end;

    if end_row >= nrows ||
        end_col >= ncols ||
        start_row >= end_row ||
        start_col >= end_col ||
        row >= end_row - start_row ||
        col >= end_col - start_col {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |i, j| (i, j));
    let v = m.view(start, end);
    let i = &index;

    TestResult::from_bool(v.index(i) == &(start_row + row, start_col + col))
}
