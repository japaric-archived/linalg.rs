#![feature(globs, macro_rules, phase)]

extern crate linalg;
extern crate quickcheck;
#[phase(plugin)]
extern crate quickcheck_macros;

mod setup;

mod col {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `iter_mut().rev()` is correct for `Col<Box<[_]>>`
    #[quickcheck]
    fn owned(size: uint) -> bool {
        setup::col(size).iter_mut().rev().enumerate().all(|(i, &e)| {
            let i = size - i - 1;

            e == i
        })
    }

    // Test that `iter_mut().rev()` is correct for `Col<&mut [_]>`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (uint, uint), col: uint) -> TestResult {
        enforce! {
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let n = m.nrows();
            let mut c = try!(m.col_mut(col));

            c.iter_mut().rev().enumerate().all(|(i, &e)| {
                let i = n - i - 1;

                e == (i, col)
            })
        })
    }

    // Test that `iter_mut().rev()` is correct for `Col<strided::MutSlice>`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (uint, uint), col: uint) -> TestResult {
        enforce! {
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let n = m.nrows();
            let mut c = try!(m.col_mut(col));

            c.iter_mut().rev().enumerate().all(|(i, &e)| {
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

    // Test that `iter_mut().rev()` is correct for `Diag<strided::MutSlice>`
    #[quickcheck]
    fn strided_mut(size: (uint, uint), diag: int) -> TestResult {
        validate_diag!(diag, size);

        test!({
            let mut m = setup::mat(size);
            let mut d = try!(m.diag_mut(diag));
            let n = d.len();

            if diag > 0 {
                d.iter_mut().rev().enumerate().all(|(i, &e)| {
                    let i = n - i - 1;

                    e == (i, i + diag as uint)
                })
            } else {
                d.iter_mut().rev().enumerate().all(|(i, &e)| {
                    let i = n - i - 1;

                    e == (i - diag as uint, i)
                })
            }
        })
    }
}

mod row {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `iter_mut().rev()` is correct for `Row<Box<[_]>>`
    #[quickcheck]
    fn owned(size: uint) -> bool {
        setup::row(size).iter_mut().rev().enumerate().all(|(i, &e)| {
            let i = size - i - 1;

            e == i
        })
    }

    // Test that `iter_mut().rev()` is correct for `Row<&mut [_]>`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce! {
            row < nrows,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let n = m.ncols();
            let mut r = try!(m.row_mut(row));

            r.iter_mut().rev().enumerate().all(|(i, &e)| {
                let i = n - i - 1;

                e == (i, row)
            })
        })
    }

    // Test that `iter_mut().rev()` is correct for `Row<strided::MutSlice>`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce! {
            row < nrows,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let n = m.ncols();
            let mut r = try!(m.row_mut(row));

            r.iter_mut().rev().enumerate().all(|(i, &e)| {
                let i = n - i - 1;

                e == (row, i)
            })
        })
    }
}
