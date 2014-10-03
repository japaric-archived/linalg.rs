use notsafe::UnsafeMatrixMutRow;
use strided;
use traits::{Matrix, MatrixMutRow};
use {MutView, Row};

impl<'a, 'b, T> UnsafeMatrixMutRow<'b, strided::MutSlice<'b, T>> for MutView<'a, T> {
    unsafe fn unsafe_mut_row(&'b mut self, row: uint) -> Row<strided::MutSlice<'b, T>> {
        let len = self.ncols();
        let stride = self.stride;

        let data = strided::MutSlice::new(self.data.offset(row as int), len, stride);

        Row(data)
    }
}

impl<'a, 'b, T> MatrixMutRow<'b, strided::MutSlice<'b, T>> for MutView<'a, T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixMutRow, MutIter, OptionMutSlice, OptionIndex, OptionIndexMut};

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_mut().and_then(|v| v.mut_row(row)).as_ref().and_then(|r| r.at(&col)) {
            let (start_row, start_col) = start;
            let col_ = start_col + col;
            let row_ = start_row + row;

            TestResult::from_bool((row_, col_).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn at_mut(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_mut().and_then(|v| v.mut_row(row)).as_mut().and_then(|r| r.at_mut(&col)) {
            let (start_row, start_col) = start;
            let col_ = start_col + col;
            let row_ = start_row + row;

            TestResult::from_bool((row_, col_).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        row: uint,
    ) -> TestResult {
        if let Some(r) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_mut().and_then(|v| v.mut_row(row)) {
            let (start_row, start_col) = start;

            TestResult::from_bool(r.iter().enumerate().all(|(col, e)| {
                e.eq(&(start_row + row, start_col + col))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        row: uint,
    ) -> TestResult {
        if let Some(mut r) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_mut().and_then(|v| v.mut_row(row)) {
            let (start_row, start_col) = start;

            TestResult::from_bool(r.mut_iter().enumerate().all(|(col, e)| {
                e.eq(&&mut (start_row + row, start_col + col))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, skip): (uint, uint),
    ) -> TestResult {
        if let Some(mut r) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_mut().and_then(|v| v.mut_row(row)) {
            let ncols = test::size(start, end).1;

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
    fn rev_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        row: uint,
    ) -> TestResult {
        if let Some(r) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_mut().and_then(|v| v.mut_row(row)) {
            let ncols = test::size(start, end).1;
            let (start_row, start_col) = start;

            TestResult::from_bool(r.iter().rev().enumerate().all(|(col, e)| {
                e.eq(&(start_row + row, start_col + ncols - col - 1))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_mut_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        row: uint
    ) -> TestResult {
        if let Some(mut r) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_mut().and_then(|v| v.mut_row(row)) {
            let ncols = test::size(start, end).1;
            let (start_row, start_col) = start;

            TestResult::from_bool(r.mut_iter().rev().enumerate().all(|(col, e)| {
                e.eq(&&mut (start_row + row, start_col + ncols - col - 1))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, skip): (uint, uint)
    ) -> TestResult {
        if let Some(r) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_mut().and_then(|v| v.mut_row(row)) {
            let ncols = test::size(start, end).1;

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
