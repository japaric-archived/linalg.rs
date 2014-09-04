#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixCols, OptionSlice, Transpose};

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).map(|v| v.t()) {
            None => TestResult::discard(),
            Some(t) => {
                let (start_row, start_col) = start;

                TestResult::from_bool(t.cols().enumerate().all(|(col, c)| {
                    c.iter().enumerate().all(|(row, e)| {
                        e.eq(&(start_row + col, start_col + row))
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
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).map(|v| v.t()) {
            None => TestResult::discard(),
            Some(t) => {
                let (nrows, _) = test::size(start, end);

                let (start_row, start_col) = start;

                TestResult::from_bool(t.cols().rev().enumerate().all(|(col, c)| {
                    c.iter().enumerate().all(|(row, e)| {
                        e.eq(&(start_row + nrows - col - 1, start_col + row))
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
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).map(|v| v.t()) {
            None => TestResult::discard(),
            Some(t) => {
                let (nrows, _) = test::size(start, end);

                if skip < nrows {
                    let hint = t.cols().skip(skip).size_hint();

                    let left = nrows - skip;

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
                use traits::{Iter, MatrixCols, MatrixRows, OptionSlice, SumCols, Transpose};

                #[quickcheck]
                fn sum(
                    size: (uint, uint),
                    (start, end): ((uint, uint), (uint, uint)),
                    skip: uint,
                ) -> TestResult {
                    match test::rand_mat::<$ty>(size).as_ref().and_then(|m| {
                        m.slice(start, end)
                    }).map(|v| v.t()) {
                        None => TestResult::discard(),
                        Some(t) => {
                            let (nrows, _) = test::size(start, end);

                            if skip < nrows {
                                let sum = t.cols().skip(skip).sum().unwrap();

                                TestResult::from_bool(sum.iter().zip(t.rows()).all(|(&e, r)| {
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
