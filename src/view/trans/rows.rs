#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixRows, OptionSlice, Transpose};

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        if let Some(t) = test::mat(size).as_ref().and_then(|m| {
            m.slice(start, end)
        }).map(|v| v.t()) {
            let (start_row, start_col) = start;

            TestResult::from_bool(t.rows().enumerate().all(|(row, r)| {
                r.iter().enumerate().all(|(col, e)| e.eq(&(start_row + col, start_col + row)))
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
        if let Some(t) = test::mat(size).as_ref().and_then(|m| {
            m.slice(start, end)
        }).map(|v| v.t()) {
            let ncols = test::size(start, end).1;
            let (start_row, start_col) = start;

            TestResult::from_bool(t.rows().rev().enumerate().all(|(row, r)| {
                r.iter().enumerate().all(|(col, e)| {
                    e.eq(&(start_row + col, start_col + ncols - row - 1))
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
        if let Some(t) = test::mat(size).as_ref().and_then(|m| {
            m.slice(start, end)
        }).map(|v| v.t()) {
            let ncols = test::size(start, end).1;

            if skip < ncols {
                let hint = t.rows().skip(skip).size_hint();

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
                use traits::{Iter, MatrixCols, MatrixRows, OptionSlice, SumRows, Transpose};

                #[quickcheck]
                fn sum(
                    size: (uint, uint),
                    (start, end): ((uint, uint), (uint, uint)),
                    skip: uint
                ) -> TestResult {
                    if let Some(t) = test::rand_mat::<$ty>(size).as_ref().and_then(|m| {
                        m.slice(start, end)
                    }).map(|v| v.t()) {
                        let ncols = test::size(start, end).1;

                        if skip < ncols {
                            let sum = t.rows().skip(skip).sum().unwrap();

                            TestResult::from_bool(sum.iter().zip(t.cols()).all(|(&e, c)| {
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
