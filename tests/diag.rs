#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate linalg;
extern crate quickcheck;
extern crate rand;

use linalg::prelude::*;
use quickcheck::TestResult;

#[macro_use]
mod setup;

mod trans {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `diag(_)` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat((nrows, ncols): (usize, usize), (diag, idx): (isize, usize)) -> TestResult {
        validate_diag_index!(diag, (nrows, ncols), idx);

        test!({
            let m = setup::mat((ncols, nrows)).t();
            let d = try!(m.diag(diag));
            let &e = try!(d.at(idx));

            e == if diag > 0 {
                (idx + diag as usize, idx)
            } else {
                (idx, idx - diag as usize)
            }
        })
    }

    // Test that `diag(_)` is correct for `Trans<View>`
    #[quickcheck]
    fn view(
        start: (usize, usize),
        (nrows, ncols): (usize, usize),
        (diag, idx): (isize, usize),
    ) -> TestResult {
        validate_diag_index!(diag, (nrows, ncols), idx);

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let m = setup::mat(size);
            let v = try!(m.slice(start..)).t();
            let d = try!(v.diag(diag));
            let &e = try!(d.at(idx));
            let (start_row, start_col) = start;
            let diag = -diag;

            e == if diag > 0 {
                (start_row + idx, start_col + idx + diag as usize)
            } else {
                (start_row + idx - diag as usize, start_col + idx)
            }
        })
    }

    // Test that `diag(_)` is correct for `Trans<MutView>`
    #[quickcheck]
    fn view_mut(
        start: (usize, usize),
        (nrows, ncols): (usize, usize),
        (diag, idx): (isize, usize),
    ) -> TestResult {
        validate_diag_index!(diag, (nrows, ncols), idx);

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let mut m = setup::mat(size);
            let v = try!(m.slice_mut(start..)).t();
            let d = try!(v.diag(diag));
            let &e = try!(d.at(idx));
            let (start_row, start_col) = start;
            let diag = -diag;

            e == if diag > 0 {
                (start_row + idx, start_col + idx + diag as usize)
            } else {
                (start_row + idx - diag as usize, start_col + idx)
            }
        })
    }
}

// Test that `diag(_)` is correct for `Mat`
#[quickcheck]
fn mat(size: (usize, usize), (diag, idx): (isize, usize)) -> TestResult {
    validate_diag_index!(diag, size, idx);

    test!({
        let m = setup::mat(size);
        let d = try!(m.diag(diag));
        let &e = try!(d.at(idx));

        e == if diag > 0 {
            (idx, idx + diag as usize)
        } else {
            (idx - diag as usize, idx)
        }
    })
}

// Test that `diag(_)` is correct for `View`
#[quickcheck]
fn view(
    start: (usize, usize),
    (nrows, ncols): (usize, usize),
    (diag, idx): (isize, usize),
) -> TestResult {
    validate_diag_index!(diag, (nrows, ncols), idx);

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let m = setup::mat(size);
        let v = try!(m.slice(start..));
        let d = try!(v.diag(diag));
        let &e = try!(d.at(idx));
        let (start_row, start_col) = start;

        e == if diag > 0 {
            (start_row + idx, start_col + idx + diag as usize)
        } else {
            (start_row + idx - diag as usize, start_col + idx)
        }
    })
}

// Test that `diag(_)` is correct for `MutView`
#[quickcheck]
fn view_mut(
    start: (usize, usize),
    (nrows, ncols): (usize, usize),
    (diag, idx): (isize, usize),
) -> TestResult {
    validate_diag_index!(diag, (nrows, ncols), idx);

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let mut m = setup::mat(size);
        let v = try!(m.slice_mut(start..));
        let d = try!(v.diag(diag));
        let &e = try!(d.at(idx));
        let (start_row, start_col) = start;

        e == if diag > 0 {
            (start_row + idx, start_col + idx + diag as usize)
        } else {
            (start_row + idx - diag as usize, start_col + idx)
        }
    })
}
