#![feature(globs, macro_rules, phase)]

extern crate linalg;
extern crate quickcheck;
#[phase(plugin)]
extern crate quickcheck_macros;

use linalg::prelude::*;
use quickcheck::TestResult;

mod setup;
mod utils;

mod trans {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `diag_mut(_)` is correct for `Trans<Mat>`
    #[quickcheck]
    fn mat((nrows, ncols): (uint, uint), (diag, idx): (int, uint)) -> TestResult {
        validate_diag_index!(diag, (nrows, ncols), idx)

        test!({
            let mut m = setup::mat((ncols, nrows)).t();
            let d = try!(m.diag_mut(diag));
            let &e = try!(d.at(idx));

            e == if diag > 0 {
                (idx + diag as uint, idx)
            } else {
                (idx, idx - diag as uint)
            }
        })
    }

    // Test that `diag_mut(_)` is correct for `Trans<MutView>`
    #[quickcheck]
    fn view_mut(
        start: (uint, uint),
        (nrows, ncols): (uint, uint),
        (diag, idx): (int, uint),
    ) -> TestResult {
        validate_diag_index!(diag, (nrows, ncols), idx)

        let size = (start.0 + ncols, start.1 + nrows);
        test!({
            let mut m = setup::mat(size);
            let mut v = try!(m.slice_from_mut(start)).t();
            let d = try!(v.diag_mut(diag));
            let &e = try!(d.at(idx));
            let (start_row, start_col) = start;
            let diag = -diag;

            e == if diag > 0 {
                (start_row + idx, start_col + idx + diag as uint)
            } else {
                (start_row + idx - diag as uint, start_col + idx)
            }
        })
    }
}

// Test that `diag_mut(_)` is correct for `Mat`
#[quickcheck]
fn mat(size: (uint, uint), (diag, idx): (int, uint)) -> TestResult {
    validate_diag_index!(diag, size, idx)

    test!({
        let mut m = setup::mat(size);
        let d = try!(m.diag_mut(diag));
        let &e = try!(d.at(idx));

        e == if diag > 0 {
            (idx, idx + diag as uint)
        } else {
            (idx - diag as uint, idx)
        }
    })
}

// Test that `diag_mut(_)` is correct for `MutView`
#[quickcheck]
fn view_mut(
    start: (uint, uint),
    (nrows, ncols): (uint, uint),
    (diag, idx): (int, uint),
) -> TestResult {
    validate_diag_index!(diag, (nrows, ncols), idx)

    let size = (start.0 + nrows, start.1 + ncols);
    test!({
        let mut m = setup::mat(size);
        let mut v = try!(m.slice_from_mut(start));
        let d = try!(v.diag_mut(diag));
        let &e = try!(d.at(idx));
        let (start_row, start_col) = start;

        e == if diag > 0 {
            (start_row + idx, start_col + idx + diag as uint)
        } else {
            (start_row + idx - diag as uint, start_col + idx)
        }
    })
}
