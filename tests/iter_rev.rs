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

    // Test that `iter().rev()` is correct for `Col<Box<[_]>>`
    #[quickcheck]
    fn owned(size: uint) -> bool {
        setup::col(size).iter().rev().enumerate().all(|(i, &e)| {
            let i = size - i - 1;

            e == i
        })
    }

    // Test that `iter().rev()` is correct for `Col<&[_]>`
    #[quickcheck]
    fn slice((nrows, ncols): (uint, uint), col: uint) -> TestResult {
        enforce!{
            col < ncols,
        }

        test!({
            let m = setup::mat((nrows, ncols));
            let c = try!(m.col(col));
            let n = m.nrows();

            c.iter().rev().enumerate().all(|(i, &e)| {
                let i = n - i - 1;

                e == (i, col)
            })
        })
    }

    // Test that `iter().rev()` is correct for `Col<&mut [_]>`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (uint, uint), col: uint) -> TestResult {
        enforce!{
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let n = m.nrows();
            let c = try!(m.col_mut(col));

            c.iter().rev().enumerate().all(|(i, &e)| {
                let i = n - i - 1;

                e == (i, col)
            })
        })
    }

    // Test that `iter().rev()` is correct for `Col<strided::Slice>`
    #[quickcheck]
    fn strided((nrows, ncols): (uint, uint), col: uint) -> TestResult {
        enforce!{
            col < ncols,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let c = try!(m.col(col));
            let n = m.nrows();

            c.iter().rev().enumerate().all(|(i, &e)| {
                let i = n - i - 1;

                e == (col, i)
            })
        })
    }

    // Test that `iter().rev()` is correct for `Col<strided::MutSlice>`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (uint, uint), col: uint) -> TestResult {
        enforce!{
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let n = m.nrows();
            let c = try!(m.col_mut(col));

            c.iter().rev().enumerate().all(|(i, &e)| {
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

    // Test that `iter().rev()` is correct for `Diag<strided::Slice>`
    #[quickcheck]
    fn strided(size: (uint, uint), diag: int) -> TestResult {
        validate_diag!(diag, size);

        test!({
            let m = setup::mat(size);
            let d = try!(m.diag(diag));
            let n = d.len();

            if diag > 0 {
                d.iter().rev().enumerate().all(|(i, &e)| {
                    let i = n - i - 1;

                    e == (i, i + diag as uint)
                })
            } else {
                d.iter().rev().enumerate().all(|(i, &e)| {
                    let i = n - i - 1;

                    e == (i - diag as uint, i)
                })
            }
        })
    }

    // Test that `iter().rev()` is correct for `Diag<strided::MutSlice>`
    #[quickcheck]
    fn strided_mut(size: (uint, uint), diag: int) -> TestResult {
        validate_diag!(diag, size);

        test!({
            let mut m = setup::mat(size);
            let d = try!(m.diag_mut(diag));
            let n = d.len();

            if diag > 0 {
                d.iter().rev().enumerate().all(|(i, &e)| {
                    let i = n - i - 1;

                    e == (i, i + diag as uint)
                })
            } else {
                d.iter().rev().enumerate().all(|(i, &e)| {
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

    // Test that `iter().rev()` is correct for `Row<Box<[_]>>`
    #[quickcheck]
    fn owned(size: uint) -> bool {
        setup::row(size).iter().rev().enumerate().all(|(i, &e)| {
            let i = size - i - 1;

            e == i
        })
    }

    // Test that `iter().rev()` is correct for `Row<&[_]>`
    #[quickcheck]
    fn slice((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce!{
            row < nrows,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let r = try!(m.row(row));
            let n = m.ncols();

            r.iter().rev().enumerate().all(|(i, &e)| {
                let i = n - i - 1;

                e == (i, row)
            })
        })
    }

    // Test that `iter().rev()` is correct for `Row<&mut [_]>`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce!{
            row < nrows,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let n = m.ncols();
            let r = try!(m.row_mut(row));

            r.iter().rev().enumerate().all(|(i, &e)| {
                let i = n - i - 1;

                e == (i, row)
            })
        })
    }

    // Test that `iter().rev()` is correct for `Row<strided::Slice>`
    #[quickcheck]
    fn strided((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce!{
            row < nrows,
        }

        test!({
            let m = setup::mat((nrows, ncols));
            let r = try!(m.row(row));
            let n = m.ncols();

            r.iter().rev().enumerate().all(|(i, &e)| {
                let i = n - i - 1;

                e == (row, i)
            })
        })
    }

    // Test that `iter().rev()` is correct for `Row<strided::MutSlice>`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce!{
            row < nrows,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let n = m.ncols();
            let r = try!(m.row_mut(row));

            r.iter().rev().enumerate().all(|(i, &e)| {
                let i = n - i - 1;

                e == (row, i)
            })
        })
    }
}
