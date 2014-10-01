use Mat;
use traits::MatrixCols;

impl<'a, T> MatrixCols<'a> for Mat<T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixCols};

    #[quickcheck]
    fn iter(size: (uint, uint)) -> TestResult {
        if let Some(m) = test::mat(size) {
            TestResult::from_bool(m.cols().enumerate().all(|(col, c)| {
                c.iter().enumerate().all(|(row, e)| e.eq(&(row, col)))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint)) -> TestResult {
        if let Some(m) = test::mat(size) {
            let ncols = size.1;

            TestResult::from_bool(m.cols().rev().enumerate().all(|(col, c)| {
                c.iter().enumerate().all(|(row, e)| e.eq(&(row, ncols - col - 1)))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), skip: uint) -> TestResult {
        if let Some(m) = test::mat(size) {
            let ncols = size.1;

            if skip < ncols {
                let hint = m.cols().skip(skip).size_hint();

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
                use traits::{Iter, MatrixCols, MatrixRows, SumCols};

                #[quickcheck]
                fn sum(size: (uint, uint), skip: uint) -> TestResult {
                    if let Some(m) = test::rand_mat::<$ty>(size) {
                        let ncols = size.1;

                        if skip < ncols {
                            let sum = m.cols().skip(skip).sum().unwrap();

                            TestResult::from_bool(sum.iter().zip(m.rows()).all(|(&e, r)| {
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
