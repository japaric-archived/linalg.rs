use View;
use traits::MatrixRows;

impl<'a, 'b, T> MatrixRows<'b> for View<'a, T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, Matrix, MatrixRows, OptionSlice};

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint))
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)) {
            None => TestResult::discard(),
            Some(v) => {
                let (start_row, start_col) = start;

                TestResult::from_bool(v.rows().enumerate().all(|(row, r)| {
                    r.iter().enumerate().all(|(col, e)| {
                        e.eq(&(start_row + row, start_col + col))
                    })
                }))
            },
        }
    }

    #[quickcheck]
    fn rev_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint))
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)) {
            None => TestResult::discard(),
            Some(v) => {
                let (start_row, start_col) = start;

                let nrows = v.nrows();

                TestResult::from_bool(v.rows().rev().enumerate().all(|(row, r)| {
                    r.iter().enumerate().all(|(col, e)| {
                        e.eq(&(start_row + nrows - row - 1, start_col + col))
                    })
                }))
            },
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        skip: uint
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)) {
            None => TestResult::discard(),
            Some(v) => {
                let nrows = v.nrows();

                if skip < nrows {
                    let hint = v.rows().skip(skip).size_hint();

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
                use traits::{Iter, MatrixCols, MatrixRows, OptionSlice, SumRows};

                #[quickcheck]
                fn sum(
                    size: (uint, uint),
                    (start, end): ((uint, uint), (uint, uint)),
                    skip: uint
                ) -> TestResult {
                    match test::rand_mat::<$ty>(size).as_ref().and_then(|m| m.slice(start, end)) {
                        None => TestResult::discard(),
                        Some(v) => {
                            let (nrows, _) = test::size(start, end);

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
                        }
                    }
                }
            }
        )+}
    }

    sum!(f32, f64)
}
