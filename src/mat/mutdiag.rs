use std::cmp;

use strided;
use traits::{Matrix, MatrixMutDiag};
use {Diag, Mat};

impl<T> MatrixMutDiag<T> for Mat<T> {
    fn mut_diag(&mut self, diag: int) -> Option<Diag<strided::MutSlice<T>>> {
        let (nrows, ncols) = self.size();
        let stride = self.stride;

        if diag > 0 {
            let diag = diag as uint;

            if diag < ncols {
                let ptr = unsafe { self.data.as_mut_ptr().offset((diag * stride) as int) };
                let len = cmp::min(nrows, ncols - diag);

                Some(Diag {
                    data: strided::MutSlice::new(ptr, len, stride + 1),
                })
            } else {
                None
            }
        } else {
            let diag = -diag as uint;

            if diag < nrows {
                let ptr = unsafe { self.data.as_mut_ptr().offset(diag as int) };
                let len = cmp::min(nrows - diag, ncols);

                Some(Diag {
                    data: strided::MutSlice::new(ptr, len, stride + 1),
                })
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixMutDiag, MutIter, OptionIndex, OptionIndexMut};

    #[quickcheck]
    fn at(size: (uint, uint), diag: int, index: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_diag(diag)
        }).as_ref().and_then(|d| d.at(&index)) {
            None => TestResult::discard(),
            Some(e) => {
                let (row, col) = if diag > 0 {
                    (index, index + diag as uint)
                } else {
                    (index - diag as uint, index)
                };

                TestResult::from_bool((row, col).eq(e))
            },
        }
    }

    #[quickcheck]
    fn at_mut(size: (uint, uint), diag: int, index: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_diag(diag)
        }).as_mut().and_then(|d| d.at_mut(&index)) {
            None => TestResult::discard(),
            Some(e) => {
                let (row, col) = if diag > 0 {
                    (index, index + diag as uint)
                } else {
                    (index - diag as uint, index)
                };

                TestResult::from_bool((row, col).eq(e))
            },
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), diag: int) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_diag(diag)) {
            None => TestResult::discard(),
            Some(d) => {
                if diag > 0 {
                    TestResult::from_bool(d.iter().enumerate().all(|(i, e)| {
                        e.eq(&(i, i + diag as uint))
                    }))
                } else {
                    TestResult::from_bool(d.iter().enumerate().all(|(i, e)| {
                        e.eq(&(i - diag as uint, i))
                    }))
                }
            },
        }
    }

    #[quickcheck]
    fn mut_iter(size: (uint, uint), diag: int) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_diag(diag)) {
            None => TestResult::discard(),
            Some(mut d) => {
                if diag > 0 {
                    TestResult::from_bool(d.mut_iter().enumerate().all(|(i, e)| {
                        e.eq(&&mut (i, i + diag as uint))
                    }))
                } else {
                    TestResult::from_bool(d.mut_iter().enumerate().all(|(i, e)| {
                        e.eq(&&mut (i - diag as uint, i))
                    }))
                }
            },
        }
    }

    #[quickcheck]
    fn mut_size_hint(size: (uint, uint), diag: int, skip: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_diag(diag)) {
            None => TestResult::discard(),
            Some(mut d) => {
                let n = d.len();

                if skip < n {
                    let hint = d.mut_iter().skip(skip).size_hint();

                    let left = n - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), diag: int) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_diag(diag)) {
            None => TestResult::discard(),
            Some(d) => {
                let n = d.len();

                if diag > 0 {
                    TestResult::from_bool(d.iter().rev().enumerate().all(|(i, e)| {
                        e.eq(&(n - i - 1, n - i - 1 + diag as uint))
                    }))
                } else {
                    TestResult::from_bool(d.iter().rev().enumerate().all(|(i, e)| {
                        e.eq(&(n - i - 1 - diag as uint, n - i - 1))
                    }))
                }
            },
        }
    }

    #[quickcheck]
    fn rev_mut_iter(size: (uint, uint), diag: int) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_diag(diag)) {
            None => TestResult::discard(),
            Some(mut d) => {
                let n = d.len();

                if diag > 0 {
                    TestResult::from_bool(d.mut_iter().rev().enumerate().all(|(i, e)| {
                        e.eq(&&mut (n - i - 1, n - i - 1 + diag as uint))
                    }))
                } else {
                    TestResult::from_bool(d.mut_iter().rev().enumerate().all(|(i, e)| {
                        e.eq(&&mut (n - i - 1 - diag as uint, n - i - 1))
                    }))
                }
            },
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), diag: int, skip: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_diag(diag)) {
            None => TestResult::discard(),
            Some(d) => {
                let n = d.len();

                if skip < n {
                    let hint = d.iter().skip(skip).size_hint();

                    let left = n - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }
}
