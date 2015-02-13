#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate linalg;
extern crate quickcheck;
extern crate rand;

use linalg::prelude::*;
use quickcheck::TestResult;
use std::collections::BTreeSet;

#[macro_use]
mod setup;

mod col {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `iter()` is correct for `ColVec`
    #[quickcheck]
    fn owned(size: usize) -> bool {
        setup::col(size).iter().enumerate().all(|(i, &e)| e == i)
    }

    // Test that `iter()` is correct for `Col`
    #[quickcheck]
    fn slice((nrows, ncols): (usize, usize), col: usize) -> TestResult {
        enforce! {
            col < ncols,
        }

        test!({
            let m = setup::mat((nrows, ncols));
            let c = try!(m.col(col));

            c.iter().enumerate().all(|(i, &e)| e == (i, col))
        })
    }

    // Test that `iter()` is correct for `MutCol`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (usize, usize), col: usize) -> TestResult {
        enforce! {
            col < ncols,
        }

        test!({
            let mut m = setup::mat((nrows, ncols));
            let c = try!(m.col_mut(col));

            c.iter().enumerate().all(|(i, &e)| e == (i, col))
        })
    }

    // Test that `iter()` is correct for `strided::Col`
    #[quickcheck]
    fn strided((nrows, ncols): (usize, usize), col: usize) -> TestResult {
        enforce! {
            col < ncols,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let c = try!(m.col(col));

            c.iter().enumerate().all(|(i, &e)| e == (col, i))
        })
    }

    // Test that `iter()` is correct for `strided::MutCol`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (usize, usize), col: usize) -> TestResult {
        enforce! {
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

    // Test that `iter()` is correct for `Diag`
    #[quickcheck]
    fn strided(size: (usize, usize), diag: isize) -> TestResult {
        validate_diag!(diag, size);

        test!({
            let m = setup::mat(size);
            let d = try!(m.diag(diag));

            if diag > 0 {
                d.iter().enumerate().all(|(i, &e)| e == (i, i + diag as usize))
            } else {
                d.iter().enumerate().all(|(i, &e)| e == (i - diag as usize, i))
            }
        })
    }

    // Test that `iter()` is correct for `MutDiag`
    #[quickcheck]
    fn strided_mut(size: (usize, usize), diag: isize) -> TestResult {
        validate_diag!(diag, size);

        test!({
            let mut m = setup::mat(size);
            let d = try!(m.diag_mut(diag));

            if diag > 0 {
                d.iter().enumerate().all(|(i, &e)| e == (i, i + diag as usize))
            } else {
                d.iter().enumerate().all(|(i, &e)| e == (i - diag as usize, i))
            }
        })
    }
}

mod row {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `iter()` is correct for `RowVec`
    #[quickcheck]
    fn owned(size: usize) -> bool {
        setup::row(size).iter().enumerate().all(|(i, &e)| e == i)
    }

    // Test that `iter()` is correct for `Row`
    #[quickcheck]
    fn slice((nrows, ncols): (usize, usize), row: usize) -> TestResult {
        enforce! {
            row < nrows,
        }

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let r = try!(m.row(row));

            r.iter().enumerate().all(|(i, &e)| e == (i, row))
        })
    }

    // Test that `iter()` is correct for `MutRow`
    #[quickcheck]
    fn slice_mut((nrows, ncols): (usize, usize), row: usize) -> TestResult {
        enforce! {
            row < nrows,
        }

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let r = try!(m.row_mut(row));

            r.iter().enumerate().all(|(i, &e)| e == (i, row))
        })
    }

    // Test that `iter()` is correct for `strided::Row`
    #[quickcheck]
    fn strided((nrows, ncols): (usize, usize), row: usize) -> TestResult {
        enforce! {
            row < nrows,
        }

        test!({
            let m = setup::mat((nrows, ncols));
            let r = try!(m.row(row));

            r.iter().enumerate().all(|(i, &e)| e == (row, i))
        })
    }

    // Test that `iter()` is correct for `strided::MutRow`
    #[quickcheck]
    fn strided_mut((nrows, ncols): (usize, usize), row: usize) -> TestResult {
        enforce! {
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
    use std::collections::BTreeSet;

    use setup;

    // Test that `iter()` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat((nrows, ncols): (usize, usize)) -> bool {
        let mut elems = BTreeSet::new();
        for r in 0..nrows {
            for c in 0..ncols {
                elems.insert((r, c));
            }
        }

        elems == setup::mat((nrows, ncols)).t().iter().map(|&x| x).collect()
    }

    // Test that `iter()` is correct for `Trans<View>`
    #[quickcheck]
    fn view(start: (usize, usize), (nrows, ncols): (usize, usize)) -> TestResult {
        let size = (start.0 + ncols, start.1 + nrows);

        test!({
            let m = setup::mat(size);
            let v = try!(m.slice_from(start)).t();
            let (start_row, start_col) = start;

            let mut t = BTreeSet::new();
            for r in 0..nrows {
                for c in 0..ncols {
                    t.insert((start_row + c, start_col + r));
                }
            }

            t == v.iter().map(|&x| x).collect()
        })
    }

    // Test that `iter()` is correct for `Trans<MutView>`
    #[quickcheck]
    fn view_mut(start: (usize, usize), (nrows, ncols): (usize, usize)) -> TestResult {
        let size = (start.0 + ncols, start.1 + nrows);

        test!({
            let mut m = setup::mat(size);
            let v = try!(m.slice_from_mut(start)).t();
            let (start_row, start_col) = start;

            let mut t = BTreeSet::new();
            for r in 0..nrows {
                for c in 0..ncols {
                    t.insert((start_row + c, start_col + r));
                }
            }

            t == v.iter().map(|&x| x).collect()
        })
    }
}

// Test that `iter()` is correct for `Mat`
#[quickcheck]
fn mat((nrows, ncols): (usize, usize)) -> bool {
    let mut elems = BTreeSet::new();
    for r in 0..nrows {
        for c in 0..ncols {
            elems.insert((r, c));
        }
    }

    elems == setup::mat((nrows, ncols)).iter().map(|&x| x).collect()
}

// Test that `iter()` is correct for `View`
#[quickcheck]
fn view(start: (usize, usize), (nrows, ncols): (usize, usize)) -> TestResult {
    let size = (start.0 + nrows, start.1 + ncols);

    test!({
        let m = setup::mat(size);
        let v = try!(m.slice_from(start));
        let (start_row, start_col) = start;

        let mut t = BTreeSet::new();
        for r in 0..nrows {
            for c in 0..ncols {
                t.insert((start_row + r, start_col + c));
            }
        }

        t == v.iter().map(|&x| x).collect()
    })
}

// Test that `iter()` is correct for `MutView`
#[quickcheck]
fn view_mut(start: (usize, usize), (nrows, ncols): (usize, usize)) -> TestResult {
    let size = (start.0 + nrows, start.1 + ncols);

    test!({
        let mut m = setup::mat(size);
        let v = try!(m.slice_from_mut(start));
        let (start_row, start_col) = start;

        let mut t = BTreeSet::new();
        for r in 0..nrows {
            for c in 0..ncols {
                t.insert((start_row + r, start_col + c));
            }
        }

        t == v.iter().map(|&x| x).collect()
    })
}
