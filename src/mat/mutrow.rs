use strided;
use notsafe::UnsafeMatrixMutRow;
use traits::{Matrix, MatrixMutRow};
use {Mat, Row};

impl<'a, T> UnsafeMatrixMutRow<'a, strided::MutSlice<'a, T>> for Mat<T> {
    unsafe fn unsafe_mut_row(&'a mut self, row: uint) -> Row<strided::MutSlice<'a, T>> {
        let (nrows, ncols) = self.size();

        let data = strided::MutSlice::new(self.data.as_mut_ptr().offset(row as int), ncols, nrows);

        Row(data)
    }
}

impl<'a, T> MatrixMutRow<'a, strided::MutSlice<'a, T>> for Mat<T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixMutRow, MutIter, OptionIndex, OptionIndexMut};

    #[quickcheck]
    fn at(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| {
            m.mut_row(row)
        }).as_ref().and_then(|r| r.at(&col)) {
            TestResult::from_bool((row, col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn at_mut(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| {
            m.mut_row(row)
        }).as_mut().and_then(|r| r.at_mut(&col)) {
            TestResult::from_bool((row, col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), row: uint) -> TestResult {
        if let Some(r) = test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            TestResult::from_bool(r.iter().enumerate().all(|(col, e)| e.eq(&(row, col))))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_iter(size: (uint, uint), row: uint) -> TestResult {
        if let Some(mut r) = test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            TestResult::from_bool(r.mut_iter().enumerate().all(|(col, e)| e.eq(&&mut (row, col))))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_size_hint(size: (uint, uint), row: uint, skip: uint) -> TestResult {
        if let Some(mut r) = test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            let ncols = size.1;

            if skip < ncols {
                let hint = r.mut_iter().skip(skip).size_hint();

                let left = ncols - skip;

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), row: uint) -> TestResult {
        if let Some(r) = test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            let ncols = size.1;

            TestResult::from_bool(r.iter().rev().enumerate().all(|(col, e)| {
                e.eq(&(row, ncols - col - 1))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_mut_iter(size: (uint, uint), row: uint) -> TestResult {
        if let Some(mut r) = test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            let ncols = size.1;

            TestResult::from_bool(r.mut_iter().rev().enumerate().all(|(col, e)| {
                e.eq(&&mut (row, ncols - col - 1))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), row: uint, skip: uint) -> TestResult {
        if let Some(r) = test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            let ncols = size.1;

            if skip < ncols {
                let hint = r.iter().skip(skip).size_hint();

                let left = ncols - skip;

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }
}
