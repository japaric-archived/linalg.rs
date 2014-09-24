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
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_row(row)
        }).as_ref().and_then(|r| r.at(&col)) {
            None => TestResult::discard(),
            Some(e) => TestResult::from_bool((row, col).eq(e)),
        }
    }

    #[quickcheck]
    fn at_mut(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_row(row)
        }).as_mut().and_then(|r| r.at_mut(&col)) {
            None => TestResult::discard(),
            Some(e) => TestResult::from_bool((row, col).eq(e)),
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), row: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            None => TestResult::discard(),
            Some(r) => {
                TestResult::from_bool(r.iter().enumerate().all(|(col, e)| {
                    e.eq(&(row, col))
                }))
            },
        }
    }

    #[quickcheck]
    fn mut_iter(size: (uint, uint), row: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            None => TestResult::discard(),
            Some(mut r) => {
                TestResult::from_bool(r.mut_iter().enumerate().all(|(col, e)| {
                    e.eq(&&mut (row, col))
                }))
            },
        }
    }

    #[quickcheck]
    fn mut_size_hint(size: (uint, uint), row: uint, skip: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            None => TestResult::discard(),
            Some(mut r) => {
                let (_, ncols) = size;

                if skip < ncols {
                    let hint = r.mut_iter().skip(skip).size_hint();

                    let left = ncols - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), row: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            None => TestResult::discard(),
            Some(r) => {
                let (_, ncols) = size;

                TestResult::from_bool(r.iter().rev().enumerate().all(|(col, e)| {
                    e.eq(&(row, ncols - col - 1))
                }))
            },
        }
    }

    #[quickcheck]
    fn rev_mut_iter(size: (uint, uint), row: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            None => TestResult::discard(),
            Some(mut r) => {
                let (_, ncols) = size;

                TestResult::from_bool(r.mut_iter().rev().enumerate().all(|(col, e)| {
                    e.eq(&&mut (row, ncols - col - 1))
                }))
            },
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), row: uint, skip: uint) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_row(row)) {
            None => TestResult::discard(),
            Some(r) => {
                let (_, ncols) = size;

                if skip < ncols {
                    let hint = r.iter().skip(skip).size_hint();

                    let left = ncols - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }
}
