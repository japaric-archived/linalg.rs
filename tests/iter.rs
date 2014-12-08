#![feature(globs, macro_rules, phase)]

extern crate linalg;
extern crate quickcheck;
#[phase(plugin)]
extern crate quickcheck_macros;

use linalg::prelude::*;
use quickcheck::TestResult;
use std::collections::TreeSet;

mod setup;

mod col {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `iter()` is correct for `Col<Box<[_]>>`
    #[quickcheck]
    fn owned(size: uint) -> bool {
        setup::col(size).iter().enumerate().all(|(i, &e)| e == i)
    }

    // Test that `iter()` is correct for `Col<&[_]>`
    #[quickcheck]
    fn slice((nrows, ncols): (uint, uint), col: uint) -> TestResult {
        enforce!{
            col < ncols,
        }

        test!({
            let m = setup::mat((nrows, ncols));
            let c = try!(m.col(col));

            c.iter().enumerate().all(|(i, &e)| e == (i, col))
        })
    }

    // Test that `iter()` is correct for `Col<&mut [_]>`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (uint, uint), col: uint) -> TestResult {
        enforce!{
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let c = try!(m.col_mut(col));

            c.iter().enumerate().all(|(i, &e)| e == (i, col))
        })
    }

    // Test that `iter()` is correct for `Col<strided::Slice>`
    #[quickcheck]
    fn strided((nrows, ncols): (uint, uint), col: uint) -> TestResult {
        enforce!{
            col < ncols,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let c = try!(m.col(col));

            c.iter().enumerate().all(|(i, &e)| e == (col, i))
        })
    }

    // Test that `iter()` is correct for `Col<strided::MutSlice>`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (uint, uint), col: uint) -> TestResult {
        enforce!{
            col < ncols,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let c = try!(m.col_mut(col));

            c.iter().enumerate().all(|(i, &e)| e == (col, i))
        })
    }
}

mod diag {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `iter()` is correct for `Diag<strided::Slice>`
    #[quickcheck]
    fn strided(size: (uint, uint), diag: int) -> TestResult {
        validate_diag!(diag, size)

        test!({
            let m = setup::mat(size);
            let d = try!(m.diag(diag));

            if diag > 0 {
                d.iter().enumerate().all(|(i, &e)| e == (i, i + diag as uint))
            } else {
                d.iter().enumerate().all(|(i, &e)| e == (i - diag as uint, i))
            }
        })
    }

    // Test that `iter()` is correct for `Diag<strided::MutSlice>`
    #[quickcheck]
    fn strided_mut(size: (uint, uint), diag: int) -> TestResult {
        validate_diag!(diag, size)

        test!({
            let mut m = setup::mat(size);
            let d = try!(m.diag_mut(diag));

            if diag > 0 {
                d.iter().enumerate().all(|(i, &e)| e == (i, i + diag as uint))
            } else {
                d.iter().enumerate().all(|(i, &e)| e == (i - diag as uint, i))
            }
        })
    }
}

mod row {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `iter()` is correct for `Row<Box<[_]>>`
    #[quickcheck]
    fn owned(size: uint) -> bool {
        setup::row(size).iter().enumerate().all(|(i, &e)| e == i)
    }

    // Test that `iter()` is correct for `Row<&[_]>`
    #[quickcheck]
    fn slice((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce!{
            row < nrows,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let r = try!(m.row(row));

            r.iter().enumerate().all(|(i, &e)| e == (i, row))
        })
    }

    // Test that `iter()` is correct for `Row<&mut [_]>`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce!{
            row < nrows,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let r = try!(m.row_mut(row));

            r.iter().enumerate().all(|(i, &e)| e == (i, row))
        })
    }

    // Test that `iter()` is correct for `Row<strided::Slice>`
    #[quickcheck]
    fn strided((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce!{
            row < nrows,
        }

        test!({
            let m = setup::mat((nrows, ncols));
            let r = try!(m.row(row));

            r.iter().enumerate().all(|(i, &e)| e == (row, i))
        })
    }

    // Test that `iter()` is correct for `Row<strided::MutSlice>`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (uint, uint), row: uint) -> TestResult {
        enforce!{
            row < nrows,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let r = try!(m.row_mut(row));

            r.iter().enumerate().all(|(i, &e)| e == (row, i))
        })
    }
}

mod trans {
    use linalg::prelude::*;
    use quickcheck::TestResult;
    use std::collections::TreeSet;

    use setup;

    // Test that `iter()` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat((nrows, ncols): (uint, uint)) -> bool {
        let mut elems = TreeSet::new();
        for r in range(0, nrows) {
            for c in range(0, ncols) {
                elems.insert((r, c));
            }
        }

        elems == setup::mat((nrows, ncols)).t().iter().map(|&x| x).collect()
    }

    // Test that `iter()` is correct for `Trans<View>`
    #[quickcheck]
    fn view(start: (uint, uint), (nrows, ncols): (uint, uint)) -> TestResult {
        let size = (start.0 + ncols, start.1 + nrows);

        test!({
            let m = setup::mat(size);
            let v = try!(m.slice_from(start)).t();
            let (start_row, start_col) = start;

            let mut t = TreeSet::new();
            for r in range(0, nrows) {
                for c in range(0, ncols) {
                    t.insert((start_row + c, start_col + r));
                }
            }

            t == v.iter().map(|&x| x).collect()
        })
    }

    // Test that `iter()` is correct for `Trans<MutView>`
    #[quickcheck]
    fn view_mut(start: (uint, uint), (nrows, ncols): (uint, uint)) -> TestResult {
        let size = (start.0 + ncols, start.1 + nrows);

        test!({
            let mut m = setup::mat(size);
            let v = try!(m.slice_from_mut(start)).t();
            let (start_row, start_col) = start;

            let mut t = TreeSet::new();
            for r in range(0, nrows) {
                for c in range(0, ncols) {
                    t.insert((start_row + c, start_col + r));
                }
            }

            t == v.iter().map(|&x| x).collect()
        })
    }
}

// Test that `iter()` is correct for `Mat`
#[quickcheck]
fn mat((nrows, ncols): (uint, uint)) -> bool {
    let mut elems = TreeSet::new();
    for r in range(0, nrows) {
        for c in range(0, ncols) {
            elems.insert((r, c));
        }
    }

    elems == setup::mat((nrows, ncols)).iter().map(|&x| x).collect()
}

// Test that `iter()` is correct for `View`
#[quickcheck]
fn view(start: (uint, uint), (nrows, ncols): (uint, uint)) -> TestResult {
    let size = (start.0 + nrows, start.1 + ncols);

    test!({
        let m = setup::mat(size);
        let v = try!(m.slice_from(start));
        let (start_row, start_col) = start;

        let mut t = TreeSet::new();
        for r in range(0, nrows) {
            for c in range(0, ncols) {
                t.insert((start_row + r, start_col + c));
            }
        }

        t == v.iter().map(|&x| x).collect()
    })
}

// Test that `iter()` is correct for `MutView`
#[quickcheck]
fn view_mut(start: (uint, uint), (nrows, ncols): (uint, uint)) -> TestResult {
    let size = (start.0 + nrows, start.1 + ncols);

    test!({
        let mut m = setup::mat(size);
        let v = try!(m.slice_from_mut(start));
        let (start_row, start_col) = start;

        let mut t = TreeSet::new();
        for r in range(0, nrows) {
            for c in range(0, ncols) {
                t.insert((start_row + r, start_col + c));
            }
        }

        t == v.iter().map(|&x| x).collect()
    })
}
