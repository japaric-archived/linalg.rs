#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixCol, OptionIndex, Transpose};

    #[quickcheck]
    fn at(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        if let Some(e) = test::mat(size).map(|m| m.t()).as_ref().and_then(|t| {
            t.col(col)
        }).as_ref().and_then(|c| c.at(&row)) {
            TestResult::from_bool((col, row).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), col: uint) -> TestResult {
        if let Some(c) = test::mat(size).map(|m| m.t()).as_ref().and_then(|t| t.col(col)) {
            let nrows = size.0;

            if col < nrows {
                TestResult::from_bool(c.iter().enumerate().all(|(row, e)| e.eq(&(col, row))))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), col: uint) -> TestResult {
        if let Some(c) = test::mat(size).map(|m| m.t()).as_ref().and_then(|t| t.col(col)) {
            let ncols = size.1;

            TestResult::from_bool(c.iter().rev().enumerate().all(|(row, e)| {
                e.eq(&(col, ncols - row - 1))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), col: uint, skip: uint) -> TestResult {
        if let Some(c) = test::mat(size).map(|m| m.t()).as_ref().and_then(|t| t.col(col)) {
            let ncols = size.1;

            if skip < ncols {
                let hint = c.iter().skip(skip).size_hint();

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
