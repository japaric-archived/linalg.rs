use std::kinds::marker;

use {Mat, View};
use traits::{Matrix, OptionSlice};

impl<'a, T> OptionSlice<'a, (uint, uint), View<'a, T>> for Mat<T> {
    fn slice(&'a self, start: (uint, uint), end: (uint, uint)) -> Option<View<'a, T>> {
        let (end_row, end_col) = end;
        let (nrows, ncols) = self.size();
        let (start_row, start_col) = start;

        if end_col <= ncols && end_col > start_col + 1 &&
                end_row <= nrows && end_row > start_row + 1 {
            let stride = self.stride;
            let ptr = unsafe {
                self.data.as_ptr().offset((start_col * stride + start_row) as int)
            };

            Some(View {
                _contravariant: marker::ContravariantLifetime::<'a>,
                _nosend: marker::NoSend,
                data: ptr,
                size: (end_row - start_row, end_col - start_col),
                stride: stride,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;
    use std::collections::TreeSet;

    use test;
    use traits::{Iter, OptionSlice, OptionIndex};

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| {
            m.slice(start, end)
        }).as_ref().and_then(|v| v.at(&(row, col))) {
            None => TestResult::discard(),
            Some(e) => {
                let (start_row, start_col) = start;

                TestResult::from_bool((start_row + row, start_col + col).eq(e))
            },
        }
    }

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)) {
            None => TestResult::discard(),
            Some(v) => {
                let (nrows, ncols) = test::size(start, end);
                let (start_row, start_col) = start;

                let mut elems = TreeSet::new();
                for row in range(0, nrows) {
                    for col in range(0, ncols) {
                        elems.insert((start_row + row, start_col + col));
                    }
                }

                TestResult::from_bool(elems == v.iter().map(|&x| x).collect())
            },
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        skip: uint,
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)) {
            None => TestResult::discard(),
            Some(v) => {
                let total = v.len();

                if skip < total {
                    let hint = v.iter().skip(skip).size_hint();
                    let left = total - skip;

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
                use traits::{Iter, MatrixCols, MatrixRows, OptionSlice};

                #[quickcheck]
                fn mul(
                    size: (uint, uint),
                    start: (uint, uint),
                    (m, k, n): (uint, uint, uint),
                ) -> TestResult {

                    let (start_row, start_col) = start;

                    match test::rand_mat::<$ty>(size).as_ref().map(|mat| (
                        mat.slice(start, (start_row + m, start_col + k)),
                        mat.slice(start, (start_row + k, start_col + n)),
                    )) {
                        Some((Some(x), Some(y))) => {
                            let z = x * y;

                            for (r, rx) in z.rows().zip(x.rows()) {
                                for (&e, cy) in r.iter().zip(y.cols()) {
                                    let sum = rx.iter().zip(cy.iter()).map(|(x, &y)| x * y).sum();

                                    if !test::is_close(e, sum) {
                                        return TestResult::failed();
                                    }
                                }
                            }

                            TestResult::passed()
                        },
                        _ => TestResult::discard(),
                    }
                }
            }
        )+}
    }

    blas!(f32, f64)
}
