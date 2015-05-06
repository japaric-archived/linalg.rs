//! Test that immutable iterators are complete and, only for linear collections, ordered

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate cast;
extern crate linalg;
extern crate quickcheck;
extern crate rand;

use std::collections::HashSet;

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

        let mut i = 0;
        let mut iter = c.iter();

        test_eq!(iter.size_hint(), (usize::from(n - i), Some(usize::from(n - i))));
        while let Some(x) = iter.next() {
            test_eq!(x, &i);

            i += 1;

            test_eq!(iter.size_hint(), (usize::from(n - i), Some(usize::from(n - i))));
        }

        test_eq!(i, n)
    }

    #[quickcheck]
    fn contiguous((nrows, ncols): (u32, u32), col: u32) -> TestResult {
        enforce! {
            col < ncols,
        }

        let m = ::setup::mat((nrows, ncols));
        let c = m.col(col);

        let mut i = 0;
        let mut iter = c.iter();

        test_eq!(iter.size_hint(), (usize::from(nrows - i), Some(usize::from(nrows - i))));
        while let Some(x) = iter.next() {
            test_eq!(x, &(i, col));

            i += 1;

            test_eq!(iter.size_hint(), (usize::from(nrows - i), Some(usize::from(nrows - i))));
        }

        test_eq!(i, nrows)
    }

    #[quickcheck]
    fn strided((nrows, ncols): (u32, u32), col: u32) -> TestResult {
        enforce! {
            col < ncols,
        }

        let m = ::setup::mat((ncols, nrows)).t();
        let c = m.col(col);

        let mut i = 0;
        let mut iter = c.iter();

        test_eq!(iter.size_hint(), (usize::from(nrows - i), Some(usize::from(nrows - i))));
        while let Some(x) = iter.next() {
            test_eq!(x, &(col, i));

            i += 1;

            test_eq!(iter.size_hint(), (usize::from(nrows - i), Some(usize::from(nrows - i))));
        }

        test_eq!(i, nrows)
    }
}

mod row {
    use cast::From;
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn owned(n: u32) -> TestResult {
        let r = ::setup::row(n);

        let mut i = 0;
        let mut iter = r.iter();

        test_eq!(iter.size_hint(), (usize::from(n - i), Some(usize::from(n - i))));
        while let Some(x) = iter.next() {
            test_eq!(x, &i);

            i += 1;

            test_eq!(iter.size_hint(), (usize::from(n - i), Some(usize::from(n - i))));
        }

        test_eq!(i, n)
    }

    #[quickcheck]
    fn contiguous((nrows, ncols): (u32, u32), row: u32) -> TestResult {
        enforce! {
            row < nrows,
        }

        let m = ::setup::mat((nrows, ncols));
        let r = m.row(row);

        let mut i = 0;
        let mut iter = r.iter();

        test_eq!(iter.size_hint(), (usize::from(ncols - i), Some(usize::from(ncols - i))));
        while let Some(x) = iter.next() {
            test_eq!(x, &(row, i));

            i += 1;

            test_eq!(iter.size_hint(), (usize::from(ncols - i), Some(usize::from(ncols - i))));
        }

        test_eq!(i, ncols)
    }

    #[quickcheck]
    fn strided((nrows, ncols): (u32, u32), row: u32) -> TestResult {
        enforce! {
            row < nrows,
        }

        let m = ::setup::mat((ncols, nrows)).t();
        let r = m.row(row);

        let mut i = 0;
        let mut iter = r.iter();

        test_eq!(iter.size_hint(), (usize::from(ncols - i), Some(usize::from(ncols - i))));
        while let Some(x) = iter.next() {
            test_eq!(x, &(i, row));

            i += 1;

            test_eq!(iter.size_hint(), (usize::from(ncols - i), Some(usize::from(ncols - i))));
        }

        test_eq!(i, ncols)
    }
}

mod transposed {
    use std::collections::HashSet;

    use cast::From;
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn mat((nrows, ncols): (u32, u32)) -> TestResult {
        let m = ::setup::mat((nrows, ncols)).t();

        let mut elems = HashSet::new();

        for r in 0..nrows {
            for c in 0..ncols {
                elems.insert((r, c));
            }
        }

        let mut count = usize::from(nrows) * usize::from(ncols);
        let mut iter = m.iter();

        test_eq!(iter.size_hint(), (count, Some(count)));
        while let Some(x) = iter.next() {
            test!(elems.remove(x));

            count -= 1;

            test_eq!(iter.size_hint(), (count, Some(count)));
        }

        test!(count == 0 && elems.is_empty())
    }

    #[quickcheck]
    fn submat((srow, scol): (u32, u32), (nrows, ncols): (u32, u32)) -> TestResult {
        let m = ::setup::mat((srow + nrows, scol + ncols));
        let v = m.slice((srow.., scol..)).t();

        let mut elems = HashSet::new();

        for r in 0..nrows {
            for c in 0..ncols {
                elems.insert((srow + r, scol + c));
            }
        }

        let mut count = usize::from(nrows) * usize::from(ncols);
        let mut iter = v.iter();

        test_eq!(iter.size_hint(), (count, Some(count)));
        while let Some(x) = iter.next() {
            test!(elems.remove(x));

            count -= 1;

            test_eq!(iter.size_hint(), (count, Some(count)));
        }

        test!(count == 0 && elems.is_empty())
    }
}

#[quickcheck]
fn diag((nrows, ncols): (u32, u32), i: i32) -> TestResult {
    let n = validate_diag_index!((nrows, ncols), i, 0);

    let m = ::setup::mat((nrows, ncols));
    let d = m.diag(i);

    let j = if i > 0 {
        let i = u32::from(i).unwrap();

        let mut j = 0;
        let mut iter = d.iter();

        test_eq!(iter.size_hint(), (usize::from(n - j), Some(usize::from(n - j ))));
        while let Some(x) = iter.next() {
            test_eq!(x, &(j, i + j));

            j += 1;

            test_eq!(iter.size_hint(), (usize::from(n - j), Some(usize::from(n - j ))));
        }

        j
    } else {
        let i = u32::from(-i).unwrap();

        let mut j = 0;
        let mut iter = d.iter();

        test_eq!(iter.size_hint(), (usize::from(n - j), Some(usize::from(n - j ))));
        while let Some(x) = iter.next() {
            test_eq!(x, &(i + j, j));

            j += 1;

            test_eq!(iter.size_hint(), (usize::from(n - j), Some(usize::from(n - j ))));
        }

        j
    };

    test_eq!(j, n)
}

#[quickcheck]
fn mat((nrows, ncols): (u32, u32)) -> TestResult {
    let m = ::setup::mat((nrows, ncols));

    let mut elems = HashSet::new();

    for r in 0..nrows {
        for c in 0..ncols {
            elems.insert((r, c));
        }
    }

    let mut count = usize::from(nrows) * usize::from(ncols);
    let mut iter = m.iter();

    test_eq!(iter.size_hint(), (count, Some(count)));
    while let Some(x) = iter.next() {
        test!(elems.remove(x));

        count -= 1;

        test_eq!(iter.size_hint(), (count, Some(count)));
    }

    test!(count == 0 && elems.is_empty())
}

#[quickcheck]
fn submat((srow, scol): (u32, u32), (nrows, ncols): (u32, u32)) -> TestResult {
    let m = ::setup::mat((srow + nrows, scol + ncols));
    let v = m.slice((srow.., scol..));

    let mut elems = HashSet::new();

    for r in 0..nrows {
        for c in 0..ncols {
            elems.insert((srow + r, scol + c));
        }
    }

    let mut count = usize::from(nrows) * usize::from(ncols);
    let mut iter = v.iter();

    test_eq!(iter.size_hint(), (count, Some(count)));
    while let Some(x) = iter.next() {
        test!(elems.remove(x));

        count -= 1;

        test_eq!(iter.size_hint(), (count, Some(count)));
    }

    test!(count == 0 && elems.is_empty())
}
