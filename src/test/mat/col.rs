use quickcheck::{TestResult,quickcheck};

use mat;
use mat::traits::{MatrixCol,MatrixColIterator};
// FIXME mozilla/rust#6515 Use std Index
use traits::{Index,Iterable};

#[test]
fn index() {
    fn prop(shape@(nrows, ncols): (uint, uint),
            col: uint)
            -> TestResult {
        if col >= ncols {
            return TestResult::discard();
        }

        let m = mat::from_fn(shape, |r, c| (r, c));
        let m_col = m.col(col);
        let mut rows = range(0, nrows);

        TestResult::from_bool(rows.all(|row| {
            m_col.index(&row).eq(&(row, col))
        }))
    }

    quickcheck(prop);
}

#[test]
fn iterable() {
    fn prop(shape@(_, ncols): (uint, uint),
            col: uint)
            -> TestResult {
        if col >= ncols {
            return TestResult::discard();
        }

        let m = mat::from_fn(shape, |r, c| (r, c));
        let m_col = m.col(col);

        TestResult::from_bool(m_col.iter().enumerate().all(|(r, e)| {
            m.index(&(r, col)).eq(e)
        }))
    }

    quickcheck(prop);
}

#[test]
fn iterator() {
    fn prop(shape: (uint, uint)) -> bool {
        let m = mat::from_fn(shape, |r, c| (r, c));

        m.cols().enumerate().all(|(c, col)| {
            col.iter().enumerate().all(|(r, e)| {
                m.index(&(r, c)).eq(e)
            })
        })
    }

    quickcheck(prop);
}
