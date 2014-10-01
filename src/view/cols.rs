use View;
use traits::MatrixCols;

impl<'a, 'b, T> MatrixCols<'b> for View<'a, T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixCols, OptionSlice};

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        if let Some(v) = test::mat(size).as_ref().and_then(|m| m.slice(start, end)) {
            let (start_row, start_col) = start;

            TestResult::from_bool(v.cols().enumerate().all(|(col, c)| {
                c.iter().enumerate().all(|(row, e)| e.eq(&(start_row + row, start_col + col)))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        if let Some(v) = test::mat(size).as_ref().and_then(|m| m.slice(start, end)) {
            let (start_row, start_col) = start;
            let ncols = test::size(start, end).1;

            TestResult::from_bool(v.cols().rev().enumerate().all(|(col, c)| {
                c.iter().enumerate().all(|(row, e)| {
                    e.eq(&(start_row + row, start_col + ncols - col - 1))
                })
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        skip: uint,
    ) -> TestResult {
        if let Some(v) = test::mat(size).as_ref().and_then(|m| m.slice(start, end)) {
            let ncols = test::size(start, end).1;

            if skip < ncols {
                let hint = v.cols().skip(skip).size_hint();

                let left = ncols - skip;

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }

    macro_rules! sum {
        ($($ty:ident),+) => {$(
            mod $ty {
                use quickcheck::TestResult;
                use std::iter::AdditiveIterator as AI;

                #[allow(unused_imports)]
                use test::{c64, c128, mod};
                use traits::{Iter, MatrixCols, MatrixRows, OptionSlice, SumCols};

                #[quickcheck]
                fn sum(
                    size: (uint, uint),
                    (start, end): ((uint, uint), (uint, uint)),
                    skip: uint,
                ) -> TestResult {
                    if let Some(v) = test::rand_mat::<$ty>(size).as_ref().and_then(|m| {
                        m.slice(start, end)
                    }) {
                        let ncols = test::size(start, end).1;

                        if skip < ncols {
                            let sum = v.cols().skip(skip).sum().unwrap();

                            TestResult::from_bool(sum.iter().zip(v.rows()).all(|(&e, r)| {
                                // FIXME (rust-lang/rust#16949) Use static dispatch
                                let ai = &mut r.iter().skip(skip).map(|&x| x) as &mut AI<$ty>;
                                e == ai.sum()
                            }))
                        } else {
                            TestResult::discard()
                        }
                    } else {
                        TestResult::discard()
                    }
                }
            }
        )+}
    }

    sum!(f32, f64, c64, c128)
}
