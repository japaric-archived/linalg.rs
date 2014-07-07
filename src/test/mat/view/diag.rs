use quickcheck::TestResult;
use std::cmp;

use mat::traits::{MatrixDiag,MatrixView};
use mat;
// FIXME mozilla/rust#6515 Use std Index
use traits::Index;

#[quickcheck]
fn index(shape@(nrows, ncols): (uint, uint),
         (start, end): ((uint, uint), (uint, uint)),
         diag: int)
         -> TestResult {
    let (start_row, start_col) = start;
    let (end_row, end_col) = end;

    let v_nrows = end_row - start_row;
    let v_ncols = end_col - start_col;

    if end_row >= nrows ||
        end_col >= ncols ||
        start_row >= end_row ||
        start_col >= end_col ||
        diag <= -(v_nrows as int) ||
        diag >= v_ncols as int {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |i, j| (i, j));
    let v = m.view(start, end);
    let v_diag = v.diag(diag);

    if diag > 0 {
        let d = diag as uint;
        let n = cmp::min(v_nrows, v_ncols - d);

        TestResult::from_bool(range(0, n).all(|i| {
            v_diag.index(&i).eq(&(start_row + i, start_col + d + i))
        }))
    } else {
        let d = -diag as uint;
        let n = cmp::min(v_ncols, v_nrows - d);

        TestResult::from_bool(range(0, n).all(|i| {
            v_diag.index(&i).eq(&(start_row + d + i, start_col + i))
        }))
    }
}
