#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixRow, OptionIndex, OptionSlice, Transpose};

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).map(|v| {
            v.t()
        }).as_ref().and_then(|t| t.row(row)).as_ref().and_then(|r| r.at(&col)) {
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
        row: uint,
    ) -> TestResult {
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).map(|v| {
            v.t()
        }).as_ref().and_then(|t| t.row(row)) {
            None => TestResult::discard(),
            Some(r) => {
                let (start_row, start_col) = start;

                TestResult::from_bool(r.iter().enumerate().all(|(col, e)| {
                    e.eq(&(start_row + col, start_col + row))
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
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).map(|v| {
            v.t()
        }).as_ref().and_then(|t| t.row(row)) {
            None => TestResult::discard(),
            Some(r) => {
                let (nrows, _) = test::size(start, end);
                let (start_row, start_col) = start;

                TestResult::from_bool(r.iter().rev().enumerate().all(|(col, e)| {
                    e.eq(&(start_row + nrows - col - 1, start_col + row))
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
        match test::mat(size).as_ref().and_then(|m| m.slice(start, end)).map(|v| {
            v.t()
        }).as_ref().and_then(|t| t.row(row)) {
            None => TestResult::discard(),
            Some(r) => {
                let (nrows, _) = test::size(start, end);

                if skip < nrows {
                    let hint = r.iter().skip(skip).size_hint();

                    let left = nrows - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }
}
