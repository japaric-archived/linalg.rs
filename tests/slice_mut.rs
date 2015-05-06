//! Test that the iterators of the following slices are ordered and complete
//!
//! - m[r, s:e]
//! - m[s:e, c]
//!
//! for any valid `r`, `c`, `s`, `e`
//!
//! Test random indexing of the following slices:
//!
//! m[a:b, c:d][e:f, g:h]
//!
//! for any valid `a`, `b`, `c`, `d`, `e`, `f`, `g`, `h`

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate linalg;
extern crate quickcheck;
extern crate rand;

use linalg::prelude::*;
use quickcheck::TestResult;

#[macro_use]
mod setup;

mod transposed {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn col(
        (srow, scol): (u32, u32),
        (nrows, ncols): (u32, u32),
        (col, s, e): (u32, u32, u32),
    ) -> TestResult {
        enforce! {
            col < ncols,
            s <= e,
            e <= nrows,
        };

        let mut m = ::setup::mat((srow + ncols, scol + nrows));
        let mut v = m.slice_mut((srow.., scol..)).t();
        let c = v.slice_mut((s..e, col));

        let mut i = 0;
        for x in c {
            test_eq!(x, &mut (srow + col, scol + s + i));

            i += 1;
        }

        test_eq!(i, e - s)
    }

    #[quickcheck]
    fn submat(
        (srow, scol): (u32, u32),
        (ssrow, sscol): (u32, u32),
        (nrows, ncols): (u32, u32),
        (row, col): (u32, u32),
    ) -> TestResult {
        enforce! {
            row < nrows,
            col < ncols,
        }

        let mut m = ::setup::mat((srow + sscol + ncols, scol + ssrow + nrows));
        let mut v = m.slice_mut((srow.., scol..)).t();
        let mut vv = v.slice_mut((ssrow.., sscol..));

        test_eq!(&mut vv[(row, col)], &mut (srow + sscol + col, scol + ssrow + row))
    }

    #[quickcheck]
    fn row(
        (srow, scol): (u32, u32),
        (nrows, ncols): (u32, u32),
        (row, s, e): (u32, u32, u32),
    ) -> TestResult {
        enforce! {
            row < nrows,
            s <= e,
            e <= ncols,
        };

        let mut m = ::setup::mat((srow + ncols, scol + nrows));
        let mut v = m.slice_mut((srow.., scol..)).t();
        let r = v.slice_mut((row, s..e));

        let mut i = 0;
        for x in r {
            test_eq!(x, &mut (srow + s + i, scol + row));

            i += 1;
        }

        test_eq!(i, e - s)
    }
}

#[quickcheck]
fn col(
    (srow, scol): (u32, u32),
    (nrows, ncols): (u32, u32),
    (col, s, e): (u32, u32, u32),
) -> TestResult {
    enforce! {
        col < ncols,
        s <= e,
        e <= nrows,
    };

    let mut m = ::setup::mat((srow + nrows, scol + ncols));
    let mut v = m.slice_mut((srow.., scol..));
    let c = v.slice_mut((s..e, col));

    let mut i = 0;
    for x in c {
        test_eq!(x, &mut (srow + s + i, scol + col));

        i += 1;
    }

    test_eq!(i, e - s)
}

#[quickcheck]
fn submat(
    (srow, scol): (u32, u32),
    (ssrow, sscol): (u32, u32),
    (nrows, ncols): (u32, u32),
    (row, col): (u32, u32),
) -> TestResult {
    enforce! {
        row < nrows,
        col < ncols,
    }

    let mut m = ::setup::mat((srow + ssrow + nrows, scol + sscol + ncols));
    let mut v = m.slice_mut((srow.., scol..));
    let mut vv = v.slice_mut((ssrow.., sscol..));

    test_eq!(&mut vv[(row, col)], &mut (srow + ssrow + row, scol + sscol + col))
}

#[quickcheck]
fn row(
    (srow, scol): (u32, u32),
    (nrows, ncols): (u32, u32),
    (row, s, e): (u32, u32, u32),
) -> TestResult {
    enforce! {
        row < nrows,
        s <= e,
        e <= ncols,
    };

    let mut m = ::setup::mat((srow + nrows, scol + ncols));
    let mut v = m.slice_mut((srow.., scol..));
    let r = v.slice_mut((row, s..e));

    let mut i = 0;
    for x in r {
        test_eq!(x, &mut (srow + row, scol + s + i));

        i += 1;
    }

    test_eq!(i, e - s)
}
