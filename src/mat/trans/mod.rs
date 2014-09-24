use traits::Transpose;
use {Mat, Trans};

mod col;
mod cols;
mod diag;
mod mutcol;
mod mutcols;
mod mutdiag;
mod mutrow;
mod mutrows;
mod mutview;
mod row;
mod rows;
mod view;

impl<T> Transpose<Trans<Mat<T>>> for Mat<T> {
    fn t(self) -> Trans<Mat<T>> {
        Trans(self)
    }
}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;
    use std::collections::TreeSet;

    use test;
    use traits::{
        Iter,
        MutIter,
        OptionIndex,
        OptionIndexMut,
        Transpose,
    };

    #[quickcheck]
    fn at(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        match test::mat(size).map(|m| m.t()).as_ref().and_then(|t| t.at(&(row, col))) {
            None => TestResult::discard(),
            Some(e) => TestResult::from_bool((col, row).eq(e)),
        }
    }

    #[quickcheck]
    fn at_mut(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        match test::mat(size).map(|m| m.t()).as_mut().and_then(|t| t.at_mut(&(row, col))) {
            None => TestResult::discard(),
            Some(e) => TestResult::from_bool((col, row).eq(e)),
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint)) -> TestResult {
        match test::mat(size).map(|m| m.t()) {
            None => TestResult::discard(),
            Some(t) => {
                let (nrows, ncols) = size;

                let mut elems = TreeSet::new();
                for r in range(0, nrows) {
                    for c in range(0, ncols) {
                        elems.insert((r, c));
                    }
                }

                TestResult::from_bool(elems == t.iter().map(|&x| x).collect())
            }
        }
    }

    #[quickcheck]
    fn mut_iter(size: (uint, uint)) -> TestResult {
        match test::mat(size).map(|m| m.t()) {
            None => TestResult::discard(),
            Some(mut t) => {
                let (nrows, ncols) = size;

                let mut elems = TreeSet::new();
                for r in range(0, nrows) {
                    for c in range(0, ncols) {
                        elems.insert((r, c));
                    }
                }

                TestResult::from_bool(elems == t.mut_iter().map(|&x| x).collect())
            }
        }
    }

    #[quickcheck]
    fn mut_size_hint(
        size: (uint, uint),
        skip: uint,
    ) -> TestResult {
        match test::mat(size).map(|m| m.t()) {
            None => TestResult::discard(),
            Some(mut t) => {
                let total = t.len();

                if skip < total {
                    let left = total - skip;
                    let hint = t.mut_iter().skip(skip).size_hint();

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            }
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        skip: uint,
    ) -> TestResult {
        match test::mat(size).map(|m| m.t()) {
            None => TestResult::discard(),
            Some(t) => {
                let total = t.len();

                if skip < total {
                    let left = total - skip;
                    let hint = t.iter().skip(skip).size_hint();

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            }
        }
    }

    macro_rules! blas {
        ($($ty:ident),+) => {$(
            mod $ty {
                use quickcheck::TestResult;
                use std::iter::AdditiveIterator;

                use test;
                use traits::{Iter, MatrixCol, MatrixRow, Transpose};

                #[quickcheck]
                fn row_mul_col(length: uint, (row, col): (uint, uint)) -> TestResult {
                    match test::rand_mat::<$ty>((length, length)).map(|m| {
                        m.t()
                    }).as_ref().and_then(|t| t.row(row).and_then(|r| t.col(col).map(|c| (r, c)))) {
                        None => TestResult::discard(),
                        Some((r, c)) => {
                            TestResult::from_bool(test::is_close(
                                r * c,
                                r.iter().zip(c.iter()).map(|(x, y)| x.mul(y)).sum(),
                            ))
                        }
                    }
                }
            }
        )+}
    }

    blas!(f32, f64)
}
