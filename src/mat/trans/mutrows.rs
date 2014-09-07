#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixMutRows, Transpose};

    #[quickcheck]
    fn iter(size: (uint, uint)) -> TestResult {
        match test::mat(size).map(|m| m.t()) {
            None => TestResult::discard(),
            Some(mut t) => {
                TestResult::from_bool(t.mut_rows().enumerate().all(|(row, r)| {
                    r.iter().enumerate().all(|(col, e)| {
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
                let (_, ncols) = size;

                TestResult::from_bool(t.mut_rows().rev().enumerate().all(|(row, r)| {
                    r.iter().enumerate().all(|(col, e)| {
                        e.eq(&(col, ncols - row - 1))
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
                let (_, ncols) = size;

                if skip < ncols {
                    let hint = t.mut_rows().skip(skip).size_hint();

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

                #[allow(unused_imports)]
                use test::{c64, c128, mod};
                use traits::{Iter, MatrixCols, MatrixMutRows, SumRows, Transpose};

                #[quickcheck]
                fn sum(size: (uint, uint), skip: uint) -> TestResult {
                    match test::rand_mat::<$ty>(size).map(|m| m.t()) {
                        None => TestResult::discard(),
                        Some(mut t) => {
                            let (_, ncols) = size;

                            if skip < ncols {
                                let sum = t.mut_rows().skip(skip).sum().unwrap();

                                TestResult::from_bool(sum.iter().zip(t.cols()).all(|(&e, c)| {
                                    // FIXME (rust-lang/rust#16949) Use static dispatch
                                    let ai = &mut c.iter().skip(skip).map(|&x| x) as &mut AI<$ty>;
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

    sum!(f32, f64, c64, c128)
}
