use std::{mem, raw};

use notsafe::UnsafeMatrixCol;
use traits::{Matrix, MatrixCol};
use {Col, Mat};

impl<'a, T> UnsafeMatrixCol<'a, &'a [T]> for Mat<T> {
    unsafe fn unsafe_col(&'a self, col: uint) -> Col<&'a [T]> {
        let data = mem::transmute(raw::Slice {
            data: self.data.as_ptr().offset((col * self.stride) as int),
            len: self.nrows(),
        });

        Col(data)
    }
}

impl<'a, T> MatrixCol<'a, &'a [T]> for Mat<T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixCol, OptionIndex};

    #[quickcheck]
    fn at(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        if let Some(e) = test::mat(size).as_ref().and_then(|m| {
            m.col(col)
        }).as_ref().and_then(|c| c.at(&row)) {
            TestResult::from_bool((row, col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), col: uint) -> TestResult {
        if let Some(c) = test::mat(size).as_ref().and_then(|m| m.col(col)) {
            TestResult::from_bool(c.iter().enumerate().all(|(row, e)| e.eq(&(row, col))))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), col: uint) -> TestResult {
        if let Some(c) = test::mat(size).as_ref().and_then(|m| m.col(col)) {
            let (nrows, _) = size;

            TestResult::from_bool(c.iter().rev().enumerate().all(|(row, e)| {
                e.eq(&(nrows - row - 1, col))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), col: uint, skip: uint) -> TestResult {
        if let Some(c) = test::mat(size).as_ref().and_then(|m| m.col(col)) {
            let (nrows, _) = size;

            if skip < nrows {
                let hint = c.iter().skip(skip).size_hint();

                let left = nrows - skip;

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }
}
