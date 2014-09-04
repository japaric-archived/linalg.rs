#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixCol, OptionMutSlice, OptionIndex, Transpose};

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_ref().and_then(|t| t.col(col)).as_ref().and_then(|c| c.at(&row)) {
            None => TestResult::discard(),
            Some(e) => {
                let (start_row, start_col) = start;

                TestResult::from_bool((start_row + col, start_col + row).eq(e))
            },
        }
    }

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        col: uint,
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_ref().and_then(|t| t.col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (start_row, start_col) = start;

                TestResult::from_bool(c.iter().enumerate().all(|(row, e)| {
                    e.eq(&(start_row + col, start_col + row))
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
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_ref().and_then(|t| t.col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (_, ncols) = test::size(start, end);
                let (start_row, start_col) = start;

                TestResult::from_bool(c.iter().rev().enumerate().all(|(row, e)| {
                    e.eq(&(start_row + col, start_col + ncols - row - 1))
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
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_ref().and_then(|t| t.col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (_, ncols) = test::size(start, end);

                if skip < ncols {
                    let hint = c.iter().skip(skip).size_hint();

                    let left = ncols - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }
}
