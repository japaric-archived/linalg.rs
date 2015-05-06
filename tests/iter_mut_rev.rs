//! Test that reverse mutable iterators are ordered and complete

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
        let mut c = ::setup::col(n);

        let mut i = n;
        let mut iter = c.iter_mut().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &mut i);
        }

        test_eq!(i, 0)
    }

    #[quickcheck]
    fn contiguous((nrows, ncols): (u32, u32), col: u32) -> TestResult {
        enforce! {
            col < ncols,
        }

        let mut m = ::setup::mat((nrows, ncols));
        let mut c = m.col_mut(col);

        let mut i = nrows;
        let mut iter = c.iter_mut().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &mut (i, col));
        }

        test_eq!(i, 0)
    }

    #[quickcheck]
    fn strided((nrows, ncols): (u32, u32), col: u32) -> TestResult {
        enforce! {
            col < ncols,
        }

        let mut m = ::setup::mat((ncols, nrows)).t();
        let mut c = m.col_mut(col);

        let mut i = nrows;
        let mut iter = c.iter_mut().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &mut (col, i));
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
        let mut r = ::setup::row(n);

        let mut i = n;
        let mut iter = r.iter_mut().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &mut i);
        }

        test_eq!(i, 0)
    }

    #[quickcheck]
    fn contiguous((nrows, ncols): (u32, u32), row: u32) -> TestResult {
        enforce! {
            row < nrows,
        }

        let mut m = ::setup::mat((nrows, ncols));
        let mut r = m.row_mut(row);

        let mut i = ncols;
        let mut iter = r.iter_mut().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &mut (row, i));
        }

        test_eq!(i, 0)
    }

    #[quickcheck]
    fn strided((nrows, ncols): (u32, u32), row: u32) -> TestResult {
        enforce! {
            row < nrows,
        }

        let mut m = ::setup::mat((ncols, nrows)).t();
        let mut r = m.row_mut(row);

        let mut i = ncols;
        let mut iter = r.iter_mut().rev();

        test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
        while let Some(x) = iter.next() {
            i -= 1;

            test_eq!(iter.size_hint(), (usize::from(i), Some(usize::from(i))));
            test_eq!(x, &mut (i, row));
        }

        test_eq!(i, 0)
    }
}

#[quickcheck]
fn diag((nrows, ncols): (u32, u32), i: i32) -> TestResult {
    let n = validate_diag_index!((nrows, ncols), i, 0);

    let mut m = ::setup::mat((nrows, ncols));
    let mut d = m.diag_mut(i);

    let j = if i > 0 {
        let i = u32::from(i).unwrap();

        let mut j = n;
        let mut iter = d.iter_mut().rev();

        test_eq!(iter.size_hint(), (usize::from(j), Some(usize::from(j))));
        while let Some(x) = iter.next() {
            j -= 1;

            test_eq!(iter.size_hint(), (usize::from(j), Some(usize::from(j))));
            test_eq!(x, &mut (j, i + j));
        }

        j
    } else {
        let i = u32::from(-i).unwrap();

        let mut j = n;
        let mut iter = d.iter_mut().rev();

        test_eq!(iter.size_hint(), (usize::from(j), Some(usize::from(j))));
        while let Some(x) = iter.next() {
            j -= 1;

            test_eq!(iter.size_hint(), (usize::from(j), Some(usize::from(j))));
            test_eq!(x, &mut (i + j, j));
        }

        j
    };

    test_eq!(j, 0)
}
