use rand::distributions::{IndependentSample,Range};
use std::{cmp,rand};

use array::traits::ArrayShape;
use mat::traits::{MatrixCol,MatrixColIterator,MatrixDiag,MatrixRow,
                  MatrixRowIterator,MatrixView};
use mat;
use super::rand_sizes;
use super::super::NSAMPLES;
// FIXME mozilla/rust#6515 Use std Index
use traits::{Index,Iterable};

// Index
#[test]
fn index() {
    let mut rng = rand::task_rng();

    for shape@(nrows, ncols) in rand_sizes().take(NSAMPLES) {
        let m = mat::from_fn(shape, |i, j| i - j);
        let start_row = Range::new(0, nrows / 2 - 1).ind_sample(&mut rng);
        let start_col = Range::new(0, ncols / 2 - 1).ind_sample(&mut rng);
        let stop_row = Range::new(nrows / 2, nrows).ind_sample(&mut rng);
        let stop_col = Range::new(ncols / 2, ncols).ind_sample(&mut rng);

        let v = m.view((start_row, start_col), (stop_row, stop_col));

        let view_shape@(nrows, ncols) =
            (stop_row - start_row, stop_col - start_col);

        assert_eq!(v.shape(), view_shape);

        for i in range(0, nrows) {
            for j in range(0, ncols) {
                let got = *v.index(&(i, j));
                let expected = (i + start_row) - (j + start_col);

                assert_eq!((shape, got), (shape, expected));
            }
        }
    }
}

#[test]
fn view_index() {
    let mut rng = rand::task_rng();

    for shape@(nrows, ncols) in rand_sizes().take(NSAMPLES) {
        let m = mat::from_fn(shape, |i, j| i - j);
        let offset_row = Range::new(0, nrows / 2 - 1).ind_sample(&mut rng);
        let offset_col = Range::new(0, ncols / 2 - 1).ind_sample(&mut rng);
        let stop_row = Range::new(nrows / 2, nrows).ind_sample(&mut rng);
        let stop_col = Range::new(ncols / 2, ncols).ind_sample(&mut rng);

        let v = m.view((offset_row, offset_col), (stop_row, stop_col));

        let (nrows, ncols) = (stop_row - offset_row, stop_col - offset_col);

        let mut start_row = Range::new(0, nrows / 2 - 1).ind_sample(&mut rng);
        let mut start_col = Range::new(0, ncols / 2 - 1).ind_sample(&mut rng);
        let stop_row = Range::new(nrows / 2, nrows).ind_sample(&mut rng);
        let stop_col = Range::new(ncols / 2, ncols).ind_sample(&mut rng);

        let vv = v.view((start_row, start_col), (stop_row, stop_col));

        let view_shape@(nrows, ncols) =
            (stop_row - start_row, stop_col - start_col);

        assert_eq!(vv.shape(), view_shape);

        start_row += offset_row;
        start_col += offset_col;

        for i in range(0, nrows) {
            for j in range(0, ncols) {
                let got = *vv.index(&(i, j));
                let expected = (i + start_row) - (j + start_col);

                assert_eq!((shape, got), (shape, expected));
            }
        }
    }
}

// MatrixCol
#[test]
fn col() {
    let mut rng = rand::task_rng();

    for shape@(nrows, ncols) in rand_sizes().take(NSAMPLES) {
        let m = mat::from_fn(shape, |i, j| i - j);

        let start_row = Range::new(0, nrows / 2 - 1).ind_sample(&mut rng);
        let start_col = Range::new(0, ncols / 2 - 1).ind_sample(&mut rng);
        let stop_row = Range::new(nrows / 2, nrows).ind_sample(&mut rng);
        let stop_col = Range::new(ncols / 2, ncols).ind_sample(&mut rng);

        let v = m.view((start_row, start_col), (stop_row, stop_col));

        let (nrows, ncols) = v.shape();

        for j in range(0, ncols) {
            let col = v.col(j);

            assert_eq!(col.len(), nrows);

            for i in range(0, nrows) {
                let got = *col.index(&i);
                let expected = (i + start_row) - (j + start_col);

                assert_eq!((shape, got), (shape, expected));
            }
        }
    }
}

#[test]
fn iterable_col() {
    let mut rng = rand::task_rng();

    for shape@(nrows, ncols) in rand_sizes().take(NSAMPLES) {
        let m = mat::from_fn(shape, |i, j| i - j);

        let start_row = Range::new(0, nrows / 2 - 1).ind_sample(&mut rng);
        let start_col = Range::new(0, ncols / 2 - 1).ind_sample(&mut rng);
        let stop_row = Range::new(nrows / 2, nrows).ind_sample(&mut rng);
        let stop_col = Range::new(ncols / 2, ncols).ind_sample(&mut rng);

        let v = m.view((start_row, start_col), (stop_row, stop_col));

        let (nrows, ncols) = (stop_row - start_row, stop_col - start_col);

        for j in range(0, ncols) {
            let col = v.col(j);
            let got: Vec<uint> = col.iter().map(|&x| x).collect();
            let expected = Vec::from_fn(nrows, |i| {
                (i + start_row) - (j + start_col)
            });

            assert_eq!(got, expected);
        }
    }
}

// MatrixColIterator
#[test]
fn cols() {
    let mut rng = rand::task_rng();

    for shape@(nrows, ncols) in rand_sizes().take(NSAMPLES) {
        let m = mat::from_fn(shape, |i, j| i - j);

        let start_row = Range::new(0, nrows / 2 - 1).ind_sample(&mut rng);
        let start_col = Range::new(0, ncols / 2 - 1).ind_sample(&mut rng);
        let stop_row = Range::new(nrows / 2, nrows).ind_sample(&mut rng);
        let stop_col = Range::new(ncols / 2, ncols).ind_sample(&mut rng);

        let v = m.view((start_row, start_col), (stop_row, stop_col));

        let nrows = stop_row - start_row;

        for (j, col) in v.cols().enumerate() {
            assert_eq!(col.len(), nrows);

            for i in range(0, nrows) {
                let got = *col.index(&i);
                let expected = (i + start_row) - (j + start_col);

                assert_eq!((shape, got), (shape, expected));
            }
        }
    }
}

// MatrixDiag
#[test]
fn diag() {
    let mut rng = rand::task_rng();

    for shape@(nrows, ncols) in rand_sizes().take(NSAMPLES) {
        let m = mat::from_fn(shape, |i, j| j as int - i as int);

        let start_row = Range::new(0, nrows / 2 - 1).ind_sample(&mut rng);
        let start_col = Range::new(0, ncols / 2 - 1).ind_sample(&mut rng);
        let stop_row = Range::new(nrows / 2, nrows).ind_sample(&mut rng);
        let stop_col = Range::new(ncols / 2, ncols).ind_sample(&mut rng);

        let v = m.view((start_row, start_col), (stop_row, stop_col));

        let (nrows, ncols) = v.shape();

        for d in range(-(nrows as int) + 1, ncols as int) {
            let got = v.diag(d).iter().map(|&x| x).collect();
            let expected = if d > 0 {
                Vec::from_elem(cmp::min(nrows, ncols - d as uint),
                                        d + (start_col - start_row) as int)
            } else {
                Vec::from_elem(cmp::min(nrows + d as uint, ncols),
                                        d + (start_col - start_row) as int)
            };

            assert_eq!((shape, d, got),
                       (shape, d, expected))
        }
    }
}

// MatrixRow
#[test]
fn iterable_row() {
    let mut rng = rand::task_rng();

    for shape@(nrows, ncols) in rand_sizes().take(NSAMPLES) {
        let m = mat::from_fn(shape, |i, j| i - j);

        let start_row = Range::new(0, nrows / 2 - 1).ind_sample(&mut rng);
        let start_col = Range::new(0, ncols / 2 - 1).ind_sample(&mut rng);
        let stop_row = Range::new(nrows / 2, nrows).ind_sample(&mut rng);
        let stop_col = Range::new(ncols / 2, ncols).ind_sample(&mut rng);

        let v = m.view((start_row, start_col), (stop_row, stop_col));

        let (nrows, ncols) = v.shape();

        for i in range(0, nrows) {
            let row = v.row(i);
            let got: Vec<uint> = row.iter().map(|&x| x).collect();
            let expected = Vec::from_fn(ncols, |j| {
                (i + start_row) - (j + start_col)
            });

            assert_eq!(got, expected);
        }
    }
}

#[test]
fn row() {
    let mut rng = rand::task_rng();

    for shape@(nrows, ncols) in rand_sizes().take(NSAMPLES) {
        let m = mat::from_fn(shape, |i, j| i - j);

        let start_row = Range::new(0, nrows / 2 - 1).ind_sample(&mut rng);
        let start_col = Range::new(0, ncols / 2 - 1).ind_sample(&mut rng);
        let stop_row = Range::new(nrows / 2, nrows).ind_sample(&mut rng);
        let stop_col = Range::new(ncols / 2, ncols).ind_sample(&mut rng);

        let v = m.view((start_row, start_col), (stop_row, stop_col));

        let (nrows, ncols) = v.shape();

        for i in range(0, nrows) {
            let row = v.row(i);

            assert_eq!(row.len(), ncols);

            for j in range(0, ncols) {
                let got = *row.index(&j);
                let expected = (i + start_row) - (j + start_col);

                assert_eq!((shape, got), (shape, expected));
            }
        }
    }
}

// MatrixRowIterator
#[test]
fn rows() {
    let mut rng = rand::task_rng();

    for shape@(nrows, ncols) in rand_sizes().take(NSAMPLES) {
        let m = mat::from_fn(shape, |i, j| i - j);

        let start_row = Range::new(0, nrows / 2 - 1).ind_sample(&mut rng);
        let start_col = Range::new(0, ncols / 2 - 1).ind_sample(&mut rng);
        let stop_row = Range::new(nrows / 2, nrows).ind_sample(&mut rng);
        let stop_col = Range::new(ncols / 2, ncols).ind_sample(&mut rng);

        let v = m.view((start_row, start_col), (stop_row, stop_col));

        let ncols = stop_col - start_col;

        for (i, row) in v.rows().enumerate() {
            assert_eq!(row.len(), ncols);

            for j in range(0, ncols) {
                let got = *row.index(&j);
                let expected = (i + start_row) - (j + start_col);

                assert_eq!((shape, got), (shape, expected));
            }
        }
    }
}

// MatrixView
#[test]
#[should_fail]
fn bad_indexing() {
    let m = mat::ones::<int>((10, 10));

    m.view((2, 2), (1, 1));
}

#[test]
#[should_fail]
fn col_out_of_bounds() {
    let m = mat::ones::<int>((10, 10));

    m.view((0, 0), (0, 11));
}

#[test]
#[should_fail]
fn row_out_of_bounds() {
    let m = mat::ones::<int>((10, 10));

    m.view((0, 0), (11, 0));
}

#[test]
#[should_fail]
fn view_col_out_of_bounds() {
    let m = mat::ones::<int>((10, 10));
    let v = m.view((3, 3), (6, 6));

    v.view((0, 0), (0, 4));
}

#[test]
#[should_fail]
fn view_row_out_of_bounds() {
    let m = mat::ones::<int>((10, 10));
    let v = m.view((3, 3), (6, 6));

    v.view((0, 0), (4, 0));
}
