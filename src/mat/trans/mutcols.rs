#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixMutCols, Transpose};

    #[quickcheck]
    fn iter(size: (uint, uint)) -> TestResult {
        match test::mat(size).map(|m| m.t()) {
            None => TestResult::discard(),
            Some(mut t) => {
                TestResult::from_bool(t.mut_cols().enumerate().all(|(col, c)| {
                    c.iter().enumerate().all(|(row, e)| {
                        e.eq(&(col, row))
                    })
                }))
            },
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint)) -> TestResult {
        match test::mat(size).map(|m| m.t()) {
            None => TestResult::discard(),
            Some(mut t) => {
                let (nrows, _) = size;

                TestResult::from_bool(t.mut_cols().rev().enumerate().all(|(col, c)| {
                    c.iter().enumerate().all(|(row, e)| {
                        e.eq(&(nrows - col - 1, row))
                    })
                }))
            },
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), skip: uint) -> TestResult {
        match test::mat(size).map(|m| m.t()) {
            None => TestResult::discard(),
            Some(mut t) => {
                let (nrows, _) = size;

                if skip < nrows {
                    let hint = t.mut_cols().skip(skip).size_hint();

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
                use traits::{Iter, MatrixMutCols, MatrixRows, SumCols, Transpose};

                #[quickcheck]
                fn sum(size: (uint, uint), skip: uint) -> TestResult {
                    match test::rand_mat::<$ty>(size).map(|m| m.t()) {
                        None => TestResult::discard(),
                        Some(mut t) => {
                            let (nrows, _) = size;

                            if skip < nrows {
                                let sum = t.mut_cols().skip(skip).sum().unwrap();

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
