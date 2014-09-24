use std::{mem, raw};

use notsafe::UnsafeMatrixCol;
use traits::{Matrix, MatrixCol};
use {Col, MutView};

impl<'a, 'b, T> UnsafeMatrixCol<'b, &'b [T]> for MutView<'a, T> {
    unsafe fn unsafe_col(&'b self, col: uint) -> Col<&'b [T]> {
        let data = mem::transmute(raw::Slice {
            data: self.data.offset((col * self.stride) as int) as *const T,
            len: self.nrows(),
        });

        Col(data)
    }
}

impl<'a, 'b, T> MatrixCol<'b, &'b [T]> for MutView<'a, T> {}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixCol, OptionMutSlice, OptionIndex};

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint)
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_ref().and_then(|v| v.col(col)).as_ref().and_then(|c| c.at(&row)) {
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
        col: uint,
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_ref().and_then(|v| v.col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (start_row, start_col) = start;

                TestResult::from_bool(c.iter().enumerate().all(|(row, e)| {
                    e.eq(&(start_row + row, start_col + col))
                }))
            },
        }
    }

    #[quickcheck]
    fn rev_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        col: uint,
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_ref().and_then(|v| v.col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (nrows, _) = test::size(start, end);

                let (start_row, start_col) = start;

                TestResult::from_bool(c.iter().rev().enumerate().all(|(row, e)| {
                    e.eq(&(start_row + nrows - row - 1, start_col + col))
                }))
            },
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (col, skip): (uint, uint),
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_ref().and_then(|v| v.col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (nrows, _) = test::size(start, end);

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
