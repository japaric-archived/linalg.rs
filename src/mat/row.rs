use strided;
use notsafe::UnsafeMatrixRow;
use traits::{Matrix, MatrixRow};
use {Mat, Row};

impl<'a, T> UnsafeMatrixRow<'a, strided::Slice<'a, T>> for Mat<T> {
    unsafe fn unsafe_row(&'a self, row: uint) -> Row<strided::Slice<'a, T>> {
        let (nrows, ncols) = self.size();

        let data = strided::Slice::new(self.data.as_ptr().offset(row as int), ncols, nrows);

        Row {
            data: data,
        }
    }
}

impl<'a, T> MatrixRow<'a, strided::Slice<'a, T>> for Mat<T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixRow, OptionIndex};

    #[quickcheck]
    fn at(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| {
            m.row(row)
        }).as_ref().and_then(|r| r.at(&col)) {
            None => TestResult::discard(),
            Some(e) => TestResult::from_bool((row, col).eq(e)),
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), row: uint) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.row(row)) {
            None => TestResult::discard(),
            Some(r) => {
                TestResult::from_bool(r.iter().enumerate().all(|(col, e)| {
                    e.eq(&(row, col))
                }))
            },
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), row: uint) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.row(row)) {
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
    fn size_hint(size: (uint, uint), row: uint, skip: uint) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.row(row)) {
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
