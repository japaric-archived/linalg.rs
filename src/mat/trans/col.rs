#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixCol, OptionIndex, Transpose};

    #[quickcheck]
    fn at(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        match test::mat(size).map(|m| m.t()).as_ref().and_then(|t| {
            t.col(col)
        }).as_ref().and_then(|c| c.at(&row)) {
            None => TestResult::discard(),
            Some(e) => TestResult::from_bool((col, row).eq(e)),
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), col: uint) -> TestResult {
        match test::mat(size).map(|m| m.t()).as_ref().and_then(|t| t.col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (nrows, _) = size;

                if col < nrows {
                    TestResult::from_bool(c.iter().enumerate().all(|(row, e)| {
                        e.eq(&(col, row))
                    }))
                } else {
                    TestResult::discard()
                }
            },
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), col: uint) -> TestResult {
        match test::mat(size).map(|m| m.t()).as_ref().and_then(|t| t.col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (_, ncols) = size;

                TestResult::from_bool(c.iter().rev().enumerate().all(|(row, e)| {
                    e.eq(&(col, ncols - row - 1))
                }))
            },
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), col: uint, skip: uint) -> TestResult {
        match test::mat(size).map(|m| m.t()).as_ref().and_then(|t| t.col(col)) {
            None => TestResult::discard(),
            Some(c) => {
                let (_, ncols) = size;

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
