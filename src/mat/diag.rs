use std::cmp;

use strided;
use traits::{Matrix, MatrixDiag};
use {Diag, Mat};

impl<T> MatrixDiag<T> for Mat<T> {
    fn diag(&self, diag: int) -> Option<Diag<strided::Slice<T>>> {
        let (nrows, ncols) = self.size();
        let stride = self.stride;

        if diag > 0 {
            let diag = diag as uint;

            if diag < ncols {
                let ptr = unsafe { self.data.as_ptr().offset((diag * stride) as int) };
                let len = cmp::min(nrows, ncols - diag);

                Some(Diag(strided::Slice::new(ptr, len, stride + 1)))
            } else {
                None
            }
        } else {
            let diag = -diag as uint;

            if diag < nrows {
                let ptr = unsafe { self.data.as_ptr().offset(diag as int) };
                let len = cmp::min(nrows - diag, ncols);

                Some(Diag(strided::Slice::new(ptr, len, stride + 1)))
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
    use traits::{Collection, Iter, MatrixDiag, OptionIndex};

    #[quickcheck]
    fn at(size: (uint, uint), diag: int, index: uint) -> TestResult {
        if let Some(e) = test::mat(size).as_ref().and_then(|m| {
            m.diag(diag)
        }).as_ref().and_then(|d| d.at(&index)) {
            let (row, col) = if diag > 0 {
                (index, index + diag as uint)
            } else {
                (index - diag as uint, index)
            };

            TestResult::from_bool((row, col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), diag: int) -> TestResult {
        if let Some(d) = test::mat(size).as_ref().and_then(|m| m.diag(diag)) {
            if diag > 0 {
                TestResult::from_bool(d.iter().enumerate().all(|(i, e)| {
                    e.eq(&(i, i + diag as uint))
                }))
            } else {
                TestResult::from_bool(d.iter().enumerate().all(|(i, e)| {
                    e.eq(&(i - diag as uint, i))
                }))
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), diag: int) -> TestResult {
        if let Some(d) = test::mat(size).as_ref().and_then(|m| m.diag(diag)) {
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
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), diag: int, skip: uint) -> TestResult {
        if let Some(d) = test::mat(size).as_ref().and_then(|m| m.diag(diag)) {
            let n = d.len();

            if skip < n {
                let hint = d.iter().skip(skip).size_hint();

                let left = n - skip;

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }
}
