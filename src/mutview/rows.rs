use MutView;
use traits::MatrixRows;

impl<'a, 'b, T> MatrixRows<'b> for MutView<'a, T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixRows, OptionMutSlice};

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        if let Some(v) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)) {
            let (start_row, start_col) = start;

            TestResult::from_bool(v.rows().enumerate().all(|(row, r)| {
                r.iter().enumerate().all(|(col, e)| e.eq(&(start_row + row, start_col + col)))
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
        if let Some(v) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)) {
            let (start_row, start_col) = start;
            let nrows = test::size(start, end).0;

            TestResult::from_bool(v.rows().rev().enumerate().all(|(row, r)| {
                r.iter().enumerate().all(|(col, e)| {
                    e.eq(&(start_row + nrows - row - 1, start_col + col))
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
        if let Some(v) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)) {
            let nrows = test::size(start, end).0;

            if skip < nrows {
                let hint = v.rows().skip(skip).size_hint();

                let left = nrows - skip;

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
                use traits::{Iter, MatrixCols, MatrixRows, OptionMutSlice, SumRows};

                #[quickcheck]
                fn sum(
                    size: (uint, uint),
                    (start, end): ((uint, uint), (uint, uint)),
                    skip: uint
                ) -> TestResult {
                    if let Some(v) = test::rand_mat::<$ty>(size).as_mut().and_then(|m| {
                        m.mut_slice(start, end)
                    }) {
                        let nrows = test::size(start, end).0;

                        if skip < nrows {
                            let sum = v.rows().skip(skip).sum().unwrap();

                            TestResult::from_bool(sum.iter().zip(v.cols()).all(|(&e, c)| {
                                // FIXME (rust-lang/rust#16949) Use static dispatch
                                let ai = &mut c.iter().skip(skip).map(|&x| x) as &mut AI<$ty>;
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
