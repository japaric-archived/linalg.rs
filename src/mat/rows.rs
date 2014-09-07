use Mat;
use traits::MatrixRows;

impl<'a, T> MatrixRows<'a> for Mat<T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixRows};

    #[quickcheck]
    fn iter(size: (uint, uint)) -> TestResult {
        match test::mat(size) {
            None => TestResult::discard(),
            Some(m) => {
                TestResult::from_bool(m.rows().enumerate().all(|(row, r)| {
                    r.iter().enumerate().all(|(col, e)| {
                        e.eq(&(row, col))
                    })
                }))
            },
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint)) -> TestResult {
        match test::mat(size) {
            None => TestResult::discard(),
            Some(m) => {
                let (nrows, _) = size;

                TestResult::from_bool(m.rows().rev().enumerate().all(|(row, r)| {
                    r.iter().enumerate().all(|(col, e)| {
                        e.eq(&(nrows - row - 1, col))
                    })
                }))
            },
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), skip: uint) -> TestResult {
        match test::mat(size) {
            None => TestResult::discard(),
            Some(m) => {
                let (nrows, _) = size;

                if skip < nrows {
                    let hint = m.rows().skip(skip).size_hint();

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

                #[allow(unused_imports)]
                use test::{c64, c128, mod};
                use traits::{Iter, MatrixCols, MatrixRows, SumRows};

                #[quickcheck]
                fn sum(size: (uint, uint), skip: uint) -> TestResult {
                    match test::rand_mat::<$ty>(size) {
                        None => TestResult::discard(),
                        Some(m) => {
                            let (nrows, _) = size;

                            if skip < nrows {
                                let sum = m.rows().skip(skip).sum().unwrap();

                                TestResult::from_bool(sum.iter().zip(m.cols()).all(|(&e, c)| {
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
