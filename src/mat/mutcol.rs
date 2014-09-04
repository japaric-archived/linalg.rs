use std::{mem, raw};

use notsafe::UnsafeMatrixMutCol;
use traits::MatrixMutCol;
use {Col, Mat};

impl<'a, T> UnsafeMatrixMutCol<'a, &'a mut [T]> for Mat<T> {
    unsafe fn unsafe_mut_col(&'a mut self, col: uint) -> Col<&'a mut [T]> {
        let data = mem::transmute(raw::Slice {
            data: self.data.as_ptr().offset((col * self.stride) as int),
            len: self.stride,
        });

        Col {
            data: data,
        }
    }
}

impl<'a, T> MatrixMutCol<'a, &'a mut [T]> for Mat<T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixMutCol, MutIter, OptionIndex, OptionIndexMut};

    #[quickcheck]
    fn at(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_col(col)
        }).as_ref().and_then(|c| c.at(&(row))) {
            None => TestResult::discard(),
            Some(e) => TestResult::from_bool((row, col).eq(e)),
        }
    }

    #[quickcheck]
    fn at_mut(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_col(col)
        }).as_mut().and_then(|c| c.at_mut(&row)) {
            None => TestResult::discard(),
            Some(e) => TestResult::from_bool((row, col).eq(e)),
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), col: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                TestResult::from_bool(c.iter().enumerate().all(|(row, e)| {
                    e.eq(&(row, col))
                }))
            },
        }
    }

    #[quickcheck]
    fn mut_iter(size: (uint, uint), col: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_col(col)) {
            None => TestResult::discard(),
            Some(mut c) => {
                TestResult::from_bool(c.mut_iter().enumerate().all(|(row, e)| {
                    e.eq(&&mut (row, col))
                }))
            },
        }
    }

    #[quickcheck]
    fn mut_size_hint(size: (uint, uint), col: uint, skip: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_col(col)) {
            None => TestResult::discard(),
            Some(mut c) => {
                let (nrows, _) = size;

                if skip < nrows {
                    let hint = c.mut_iter().skip(skip).size_hint();

                    let left = nrows - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), col: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (nrows, _) = size;

                TestResult::from_bool(c.iter().rev().enumerate().all(|(row, e)| {
                    e.eq(&(nrows - row - 1, col))
                }))
            },
        }
    }

    #[quickcheck]
    fn rev_mut_iter(size: (uint, uint), col: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_col(col)) {
            None => TestResult::discard(),
            Some(mut c) => {
                let (nrows, _) = size;

                TestResult::from_bool(c.mut_iter().rev().enumerate().all(|(row, e)| {
                    e.eq(&&mut (nrows - row - 1, col))
                }))
            },
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), col: uint, skip: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (nrows, _) = size;

                if skip < nrows {
                    let hint = c.iter().skip(skip).size_hint();

                    let left = nrows - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }
}
