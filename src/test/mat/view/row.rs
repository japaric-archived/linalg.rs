use quickcheck::TestResult;

use mat;
use mat::traits::{MatrixRow,MatrixRowIterator,MatrixView};
// FIXME mozilla/rust#6515 Use std Index
use traits::{Index,Iterable};

#[quickcheck]
fn index(shape@(nrows, ncols): (uint, uint),
         (start, end): ((uint, uint), (uint, uint)),
         row: uint)
         -> TestResult {
    let (start_row, start_col) = start;
    let (end_row, end_col) = end;

    if end_row >= nrows ||
        end_col >= ncols ||
        start_row >= end_row ||
        start_col >= end_col ||
        row >= end_row - start_row {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |i, j| (i, j));
    let v = m.view(start, end);
    let v_row = v.row(row);
    let mut cols = range(0, end_col - start_col);

    TestResult::from_bool(cols.all(|col| {
        v_row.index(&col).eq(&(start_row + row, start_col + col))
    }))
}

#[quickcheck]
fn iterable(shape@(nrows, ncols): (uint, uint),
            (start, end): ((uint, uint), (uint, uint)),
            row: uint)
            -> TestResult {
    let (start_row, start_col) = start;
    let (end_row, end_col) = end;

    if end_row >= nrows ||
        end_col >= ncols ||
        start_row >= end_row ||
        start_col >= end_col ||
        row >= end_row - start_row {
        return TestResult::discard();
    }

    let m = mat::from_fn(shape, |i, j| (i, j));
    let v = m.view(start, end);
    let v_row = v.row(row);

    TestResult::from_bool(v_row.iter().enumerate().all(|(c, e)| {
        v.index(&(row, c)).eq(e)
    }))
}

#[quickcheck]
fn rows(shape@(nrows, ncols): (uint, uint),
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

    TestResult::from_bool(v.rows().enumerate().all(|(r, row)| {
        row.iter().enumerate().all(|(c, e)| {
            v.index(&(r, c)).eq(e)
        })
    }))
}
