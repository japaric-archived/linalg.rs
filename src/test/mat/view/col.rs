use quickcheck::TestResult;

use mat;
use mat::traits::{MatrixCol,MatrixColIterator,MatrixView};
// FIXME mozilla/rust#6515 Use std Index
use traits::{Index,Iterable};

#[quickcheck]
fn index(shape@(nrows, ncols): (uint, uint),
         (start, end): ((uint, uint), (uint, uint)),
         col: uint)
         -> TestResult {
    let (start_row, start_col) = start;
    let (end_row, end_col) = end;

    if end_row >= nrows ||
        end_col >= ncols ||
        start_row >= end_row ||
        start_col >= end_col ||
        col >= end_col - start_col {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |i, j| (i, j));
    let v = m.view(start, end);
    let v_col = v.col(col);
    let mut rows = range(0, end_row - start_row);

    TestResult::from_bool(rows.all(|row| {
        v_col.index(&row).eq(&(start_row + row, start_col + col))
    }))
}

#[quickcheck]
fn iterable(shape@(nrows, ncols): (uint, uint),
            (start, end): ((uint, uint), (uint, uint)),
            col: uint)
            -> TestResult {
    let (start_row, start_col) = start;
    let (end_row, end_col) = end;

    if end_row >= nrows ||
        end_col >= ncols ||
        start_row >= end_row ||
        start_col >= end_col ||
        col >= end_col - start_col {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |i, j| (i, j));
    let v = m.view(start, end);
    let v_col = v.col(col);

    TestResult::from_bool(v_col.iter().enumerate().all(|(r, e)| {
        v.index(&(r, col)).eq(e)
    }))
}

#[quickcheck]
fn iterator(shape@(nrows, ncols): (uint, uint),
            (start, end): ((uint, uint), (uint, uint)))
            -> TestResult {
    let (start_row, start_col) = start;
    let (end_row, end_col) = end;

    if end_row >= nrows ||
        end_col >= ncols ||
        start_row >= end_row ||
        start_col >= end_col {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |i, j| (i, j));
    let v = m.view(start, end);

    TestResult::from_bool(v.cols().enumerate().all(|(c, col)| {
        col.iter().enumerate().all(|(r, e)| {
            v.index(&(r, c)).eq(e)
        })
    }))
}
