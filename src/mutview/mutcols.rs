use MutView;
use traits::MatrixMutCols;

impl<'a, 'b, T> MatrixMutCols<'b> for MutView<'a, T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, Matrix, MatrixMutCols, OptionMutSlice};

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)) {
            None => TestResult::discard(),
            Some(mut v) => {
                let (start_row, start_col) = start;

                TestResult::from_bool(v.mut_cols().enumerate().all(|(col, c)| {
                    c.iter().enumerate().all(|(row, e)| {
                        e.eq(&(start_row + row, start_col + col))
                    })
                }))
            },
        }
    }

    #[quickcheck]
    fn rev_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)) {
            None => TestResult::discard(),
            Some(mut v) => {
                let (start_row, start_col) = start;

                let ncols = v.ncols();

                TestResult::from_bool(v.mut_cols().rev().enumerate().all(|(col, c)| {
                    c.iter().enumerate().all(|(row, e)| {
                        e.eq(&(start_row + row, start_col + ncols - col - 1))
                    })
                }))
            },
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        skip: uint,
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)) {
            None => TestResult::discard(),
            Some(mut v) => {
                let ncols = v.ncols();

                if skip < ncols {
                    let hint = v.mut_cols().skip(skip).size_hint();

                    let left = ncols - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }

    macro_rules! sum {
        ($($ty:ident),+) => {$(
            mod $ty {
                use quickcheck::TestResult;
                use std::iter::AdditiveIterator as AI;

                use test;
                use traits::{Iter, MatrixMutCols, MatrixRows, OptionMutSlice, SumCols};

                #[quickcheck]
                fn sum(
                    size: (uint, uint),
                    (start, end): ((uint, uint), (uint, uint)),
                    skip: uint,
                ) -> TestResult {
                    match test::rand_mat::<$ty>(size).as_mut().and_then(|m| {
                        m.mut_slice(start, end)
                    }) {
                        None => TestResult::discard(),
                        Some(mut v) => {
                            let (_, ncols) = test::size(start, end);

                            if skip < ncols {
                                let sum = v.mut_cols().skip(skip).sum().unwrap();

                                TestResult::from_bool(sum.iter().zip(v.rows()).all(|(&e, r)| {
                                    // FIXME (rust-lang/rust#16949) Use static dispatch
                                    let ai = &mut r.iter().skip(skip).map(|&x| x) as &mut AI<$ty>;
                                    e == ai.sum()
                                }))
                            } else {
                                TestResult::discard()
                            }
                        }
                    }
                }
            }
        )+}
    }

    sum!(f32, f64)
}
