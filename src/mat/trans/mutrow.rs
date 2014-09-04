#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Iter, MatrixMutRow, OptionIndex, Transpose};

    #[quickcheck]
    fn at(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        match test::mat(size).map(|m| m.t()).as_mut().and_then(|t| {
            t.mut_row(row)
        }).as_ref().and_then(|r| r.at(&col)) {
            None => TestResult::discard(),
            Some(e) => TestResult::from_bool((col, row).eq(e)),
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), row: uint) -> TestResult {
        match test::mat(size).map(|m| m.t()).as_mut().and_then(|t| t.mut_row(row)) {
            None => TestResult::discard(),
            Some(r) => {
                TestResult::from_bool(r.iter().enumerate().all(|(col, e)| {
                    e.eq(&(col, row))
                }))
            },
        }
    }

    #[quickcheck]
    fn rev_iter(size: (uint, uint), row: uint) -> TestResult {
        match test::mat(size).map(|m| m.t()).as_mut().and_then(|t| t.mut_row(row)) {
            None => TestResult::discard(),
            Some(r) => {
                let (ncols, _) = size;

                TestResult::from_bool(r.iter().rev().enumerate().all(|(col, e)| {
                    e.eq(&(ncols - col - 1, row))
                }))
            },
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), row: uint, skip: uint) -> TestResult {
        match test::mat(size).map(|m| m.t()).as_mut().and_then(|t| t.mut_row(row)) {
            None => TestResult::discard(),
            Some(r) => {
                let (ncols, _) = size;

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
