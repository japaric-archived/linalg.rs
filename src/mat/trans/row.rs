#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixRow, OptionIndex, Transpose};

    #[quickcheck]
    fn at(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        if let Some(e) = test::mat(size).map(|m| m.t()).as_ref().and_then(|t| {
            t.row(row)
        }).as_ref().and_then(|r| r.at(&col)) {
            TestResult::from_bool((col, row).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), row: uint) -> TestResult {
        if let Some(r) = test::mat(size).map(|m| m.t()).as_ref().and_then(|t| t.row(row)) {
            TestResult::from_bool(r.iter().enumerate().all(|(col, e)| e.eq(&(col, row))))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), row: uint) -> TestResult {
        if let Some(r) = test::mat(size).map(|m| m.t()).as_ref().and_then(|t| t.row(row)) {
            let ncols = size.0;

            TestResult::from_bool(r.iter().rev().enumerate().all(|(col, e)| {
                e.eq(&(ncols - col - 1, row))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), row: uint, skip: uint) -> TestResult {
        if let Some(r) = test::mat(size).map(|m| m.t()).as_ref().and_then(|t| t.row(row)) {
            let ncols = size.0;

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
