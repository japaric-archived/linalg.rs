use quickcheck::TestResult;
use std::cmp;

use mat::traits::MatrixDiag;
use mat;

// FIXME rust-lang/rust#15525 Replace `index` method with `[]` operator
#[quickcheck]
fn index(shape@(nrows, ncols): (uint, uint), diag: int) -> TestResult {
    if diag <= -(nrows as int) || diag >= ncols as int {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |i, j| (i, j));
    let m_diag = m.diag(diag);

    if diag > 0 {
        let d = diag as uint;
        let n = cmp::min(nrows, ncols - d);

        TestResult::from_bool(range(0, n).all(|i| {
            m_diag.index(&i).eq(&(i, d + i))
        }))
    } else {
        let d = -diag as uint;
        let n = cmp::min(ncols, nrows - d);

        TestResult::from_bool(range(0, n).all(|i| {
            m_diag.index(&i).eq(&(d + i, i))
        }))
    }
}

// FIXME rust-lang/rust#15525 Replace `index` method with `[]` operator
#[quickcheck]
#[should_fail]
fn out_of_bounds(shape@(nrows, ncols): (uint, uint),
                 diag: int,
                 index: uint)
                 -> TestResult {
    if diag <= -(nrows as int) || diag >= ncols as int {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |i, j| (i, j));
    let m_diag = m.diag(diag);

    let n = if diag > 0 {
        cmp::min(nrows, ncols - diag as uint)
    } else {
        cmp::min(ncols, nrows + diag as uint)
    };

    if index < n {
        TestResult::discard()
    } else {
        TestResult::from_bool(m_diag.index(&index) == &(0, 0))
    }
}
