use quickcheck::TestResult;

use mat;
use mat::traits::{MatrixRow,MatrixRowIterator};
// FIXME mozilla/rust#6515 Use std Index
use traits::{Index,Iterable};

#[quickcheck]
fn index(shape@(nrows, ncols): (uint, uint), row: uint) -> TestResult {
    if row >= nrows {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |r, c| (r, c));
    let m_row = m.row(row);
    let mut cols = range(0, ncols);

    TestResult::from_bool(cols.all(|col| {
        m_row.index(&col).eq(&(row, col))
    }))
}

#[quickcheck]
fn iterable(shape@(nrows, _): (uint, uint), row: uint) -> TestResult {
    if row >= nrows {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |r, c| (r, c));
    let m_row = m.row(row);

    TestResult::from_bool(m_row.iter().enumerate().all(|(c, e)| {
        m.index(&(row, c)).eq(e)
    }))
}

#[quickcheck]
fn iterator(shape: (uint, uint)) -> bool {
    let m = mat::from_fn(shape, |r, c| (r, c));

    m.rows().enumerate().all(|(r, row)| {
        row.iter().enumerate().all(|(c, e)| {
            m.index(&(r, c)).eq(e)
        })
    })
}
