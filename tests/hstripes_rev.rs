//! `for each stripe in m.hstripes(size).rev()`
//!
//! Test that the iterator `stripe[row, :].iter()` is ordered and complete for any valid `size` and
//! `row`

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

mod transposed {
    use cast::From;
    use linalg::prelude::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn submat(
        (srow, scol): (u32, u32),
        (nrows, ncols): (u32, u32),
        (size, col): (u32, u32),
    ) -> TestResult {
        enforce! {
            size > 0,
            col < ncols,
        };

        let m = ::setup::mat((srow + ncols, scol + nrows));
        let v = m.slice((srow.., scol..)).t();

        let mut i = nrows;
        let mut stripes = v.hstripes(size).rev();
        let mut count = usize::from(nrows / size + if nrows % size == 0 { 0 } else { 1 });

        test_eq!(stripes.size_hint(), (count, Some(count)));
        while let Some(s) = stripes.next() {
            count -= 1;

            test_eq!(stripes.size_hint(), (count, Some(count)));

            for x in s.col(col).iter().rev() {
                i -= 1;

                test_eq!(x, &(srow + col, scol + i));
            }
        }

        test_eq!(i, 0)
    }

    #[quickcheck]
    fn submat_mut(
        (srow, scol): (u32, u32),
        (nrows, ncols): (u32, u32),
        (size, col): (u32, u32),
    ) -> TestResult {
        enforce! {
            size > 0,
            col < ncols,
        };

        let mut m = ::setup::mat((srow + ncols, scol + nrows));
        let mut v = m.slice_mut((srow.., scol..)).t();

        let mut i = nrows;
        let mut stripes = v.hstripes_mut(size).rev();
        let mut count = usize::from(nrows / size + if nrows % size == 0 { 0 } else { 1 });

        test_eq!(stripes.size_hint(), (count, Some(count)));
        while let Some(mut s) = stripes.next() {
            count -= 1;

            test_eq!(stripes.size_hint(), (count, Some(count)));

            for x in s.col_mut(col).iter_mut().rev() {
                i -= 1;

                test_eq!(x, &mut (srow + col, scol + i));
            }
        }

        test_eq!(i, 0)
    }
}

#[quickcheck]
fn submat(
    (srow, scol): (u32, u32),
    (nrows, ncols): (u32, u32),
    (size, col): (u32, u32),
) -> TestResult {
    enforce! {
        size > 0,
        col < ncols,
    };

    let m = ::setup::mat((srow + nrows, scol + ncols));
    let v = m.slice((srow.., scol..));

    let mut i = nrows;
    let mut stripes = v.hstripes(size).rev();
    let mut count = usize::from(nrows / size + if nrows % size == 0 { 0 } else { 1 });

    test_eq!(stripes.size_hint(), (count, Some(count)));
    while let Some(s) = stripes.next() {
        count -= 1;

        test_eq!(stripes.size_hint(), (count, Some(count)));

        for x in s.col(col).iter().rev() {
            i -= 1;

            test_eq!(x, &(srow + i, scol + col));
        }
    }

    test_eq!(i, 0)
}

#[quickcheck]
fn submat_mut(
    (srow, scol): (u32, u32),
    (nrows, ncols): (u32, u32),
    (size, col): (u32, u32),
) -> TestResult {
    enforce! {
        size > 0,
        col < ncols,
    };

    let mut m = ::setup::mat((srow + nrows, scol + ncols));
    let mut v = m.slice_mut((srow.., scol..));

    let mut i = nrows;
    let mut stripes = v.hstripes_mut(size).rev();
    let mut count = usize::from(nrows / size + if nrows % size == 0 { 0 } else { 1 });

    test_eq!(stripes.size_hint(), (count, Some(count)));
    while let Some(mut s) = stripes.next() {
        count -= 1;

        test_eq!(stripes.size_hint(), (count, Some(count)));

        for x in s.col_mut(col).iter_mut().rev() {
            i -= 1;

            test_eq!(x, &mut (srow + i, scol + col));
        }
    }

    test_eq!(i, 0)
}
