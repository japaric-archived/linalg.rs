//! Test that reverse immutable iterators are ordered and complete

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate cast;
extern crate linalg;
extern crate quickcheck;
extern crate rand;

use cast::From;
use linalg::prelude::*;
use quickcheck::TestResult;

#[macro_use]
mod setup;

mod col {
    use cast::From;
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn owned(n: u32) -> TestResult {
        let c = ::setup::col(n);

        let mut i = n;
        let mut iter = c.iter().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &i);
        }

        test_eq!(i, 0)
    }

    #[quickcheck]
    fn contiguous((nrows, ncols): (u32, u32), col: u32) -> TestResult {
        enforce! {
            col < ncols,
        }

        let m = ::setup::mat((nrows, ncols));
        let c = m.col(col);

        let mut i = nrows;
        let mut iter = c.iter().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &(i, col));
        }

        test_eq!(i, 0)
    }

    #[quickcheck]
    fn strided((nrows, ncols): (u32, u32), col: u32) -> TestResult {
        enforce! {
            col < ncols,
        }

        let m = ::setup::mat((ncols, nrows)).t();
        let c = m.col(col);

        let mut i = nrows;
        let mut iter = c.iter().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &(col, i));
        }

        test_eq!(i, 0)
    }
}

mod row {
    use cast::From;
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn owned(n: u32) -> TestResult {
        let r = ::setup::row(n);

        let mut i = n;
        let mut iter = r.iter().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &i);
        }

        test_eq!(i, 0)
    }

    #[quickcheck]
    fn contiguous((nrows, ncols): (u32, u32), row: u32) -> TestResult {
        enforce! {
            row < nrows,
        }

        let m = ::setup::mat((nrows, ncols));
        let r = m.row(row);

        let mut i = ncols;
        let mut iter = r.iter().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &(row, i));
        }

        test_eq!(i, 0)
    }

    #[quickcheck]
    fn strided((nrows, ncols): (u32, u32), row: u32) -> TestResult {
        enforce! {
            row < nrows,
        }

        let m = ::setup::mat((ncols, nrows)).t();
        let r = m.row(row);

        let mut i = ncols;
        let mut iter = r.iter().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &(i, row));
        }

        test_eq!(i, 0)
    }
}

#[quickcheck]
fn diag((nrows, ncols): (u32, u32), i: i32) -> TestResult {
    let n = validate_diag_index!((nrows, ncols), i, 0);

    let m = ::setup::mat((nrows, ncols));
    let d = m.diag(i);

    let j = if i > 0 {
        let i = u32::from(i).unwrap();

        let mut j = n;
        let mut iter = d.iter().rev();

        test_eq!(iter.size_hint(), (usize::from(j), Some(usize::from(j))));
        while let Some(x) = iter.next() {
            j -= 1;

            test_eq!(iter.size_hint(), (usize::from(j), Some(usize::from(j))));
            test_eq!(x, &(j, i + j));
        }

        j
    } else {
        let i = u32::from(-i).unwrap();

        let mut j = n;
        let mut iter = d.iter().rev();

        test_eq!(iter.size_hint(), (usize::from(j), Some(usize::from(j))));
        while let Some(x) = iter.next() {
            j -= 1;

            test_eq!(iter.size_hint(), (usize::from(j), Some(usize::from(j))));
            test_eq!(x, &(i + j, j));
        }

        j
    };

    test_eq!(j, 0)
}
