use notsafe::UnsafeMatrixRow;
use strided;
use traits::{Matrix, MatrixRow};
use {Row, View};

impl<'a, 'b, T> UnsafeMatrixRow<'b, strided::Slice<'b, T>> for View<'a, T> {
    unsafe fn unsafe_row(&'b self, row: uint) -> Row<strided::Slice<'b, T>> {
        let len = self.ncols();
        let stride = self.stride;

        let data = strided::Slice::new(self.data.offset(row as int), len, stride);

        Row {
            data: data,
        }
    }
}

impl<'a, 'b, T> MatrixRow<'b, strided::Slice<'b, T>> for View<'a, T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixRow, OptionIndex, OptionSlice};

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).as_ref().and_then(|v| {
            v.row(row)
        }).as_ref().and_then(|r| r.at(&col)) {
            None => TestResult::discard(),
            Some(e) => {
                let (start_row, start_col) = start;
                let col_ = start_col + col;
                let row_ = start_row + row;

                TestResult::from_bool((row_, col_).eq(e))
            },
        }
    }

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        row: uint,
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).as_ref().and_then(|v| {
            v.row(row)
        }) {
            None => TestResult::discard(),
            Some(r) => {
                let (start_row, start_col) = start;

                TestResult::from_bool(r.iter().enumerate().all(|(col, e)| {
                    e.eq(&(start_row + row, start_col + col))
                }))
            },
        }
    }

    #[quickcheck]
    fn rev_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        row: uint,
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).as_ref().and_then(|v| {
            v.row(row)
        }) {
            None => TestResult::discard(),
            Some(r) => {
                let (_, ncols) = test::size(start, end);

                let (start_row, start_col) = start;

                TestResult::from_bool(r.iter().rev().enumerate().all(|(col, e)| {
                    e.eq(&(start_row + row, start_col + ncols - col - 1))
                }))
            },
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, skip): (uint, uint),
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).as_ref().and_then(|v| {
            v.row(row)
        }) {
            None => TestResult::discard(),
            Some(r) => {
                let (_, ncols) = test::size(start, end);

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
