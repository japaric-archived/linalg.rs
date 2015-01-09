#![allow(unstable)]
#![feature(plugin)]

extern crate linalg;
extern crate quickcheck;
#[plugin]
extern crate quickcheck_macros;

#[macro_use]
mod setup;

mod col {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `iter_mut().rev()` is correct for `ColVec`
    #[quickcheck]
    fn owned(size: usize) -> bool {
        setup::col(size).iter_mut().rev().enumerate().all(|(i, &mut e)| {
            let i = size - i - 1;

            e == i
        })
    }

    // Test that `iter_mut().rev()` is correct for `MutCol`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (usize, usize), col: usize) -> TestResult {
        enforce! {
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let n = m.nrows();
            let mut c = try!(m.col_mut(col));

            c.iter_mut().rev().enumerate().all(|(i, &mut e)| {
                let i = n - i - 1;

                e == (i, col)
            })
        })
    }

    // Test that `iter_mut().rev()` is correct for `strided::MutCol`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (usize, usize), col: usize) -> TestResult {
        enforce! {
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let n = m.nrows();
            let mut c = try!(m.col_mut(col));

            c.iter_mut().rev().enumerate().all(|(i, &mut e)| {
                let i = n - i - 1;

                e == (col, i)
            })
        })
    }
}

mod diag {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `iter_mut().rev()` is correct for `MutDiag`
    #[quickcheck]
    fn strided_mut(size: (usize, usize), diag: isize) -> TestResult {
        validate_diag!(diag, size);

        test!({
            let mut m = setup::mat(size);
            let mut d = try!(m.diag_mut(diag));
            let n = d.len();

            if diag > 0 {
                d.iter_mut().rev().enumerate().all(|(i, &mut e)| {
                    let i = n - i - 1;

                    e == (i, i + diag as usize)
                })
            } else {
                d.iter_mut().rev().enumerate().all(|(i, &mut e)| {
                    let i = n - i - 1;

                    e == (i - diag as usize, i)
                })
            }
        })
    }
}

mod row {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `iter_mut().rev()` is correct for `RowVec`
    #[quickcheck]
    fn owned(size: usize) -> bool {
        setup::row(size).iter_mut().rev().enumerate().all(|(i, &mut e)| {
            let i = size - i - 1;

            e == i
        })
    }

    // Test that `iter_mut().rev()` is correct for `MutRow`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (usize, usize), row: usize) -> TestResult {
        enforce! {
            row < nrows,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let n = m.ncols();
            let mut r = try!(m.row_mut(row));

            r.iter_mut().rev().enumerate().all(|(i, &mut e)| {
                let i = n - i - 1;

                e == (i, row)
            })
        })
    }

    // Test that `iter_mut().rev()` is correct for `strided::MutRow`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (usize, usize), row: usize) -> TestResult {
        enforce! {
            row < nrows,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let n = m.ncols();
            let mut r = try!(m.row_mut(row));

            r.iter_mut().rev().enumerate().all(|(i, &mut e)| {
                let i = n - i - 1;

                e == (row, i)
            })
        })
    }
}
