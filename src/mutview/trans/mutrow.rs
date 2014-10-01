#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{
        Iter,
        MatrixMutRow,
        MutIter,
        OptionMutSlice,
        OptionIndex,
        OptionIndexMut,
        Transpose,
    };

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_row(row)).as_ref().and_then(|r| r.at(&col)) {
            let (start_row, start_col) = start;

            TestResult::from_bool((start_row + col, start_col + row).eq(e))
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
        if let Some(e) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_row(row)).as_mut().and_then(|r| r.at_mut(&col)) {
            let (start_row, start_col) = start;

            TestResult::from_bool((start_row + col, start_col + row).eq(e))
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
        if let Some(r) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_row(row)) {
            let (start_row, start_col) = start;

            TestResult::from_bool(r.iter().enumerate().all(|(col, e)| {
                e.eq(&(start_row + col, start_col + row))
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
        }).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_row(row)) {
            let (start_row, start_col) = start;

            TestResult::from_bool(r.mut_iter().enumerate().all(|(col, e)| {
                e.eq(&&mut (start_row + col, start_col + row))
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
        }).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_row(row)) {
            let nrows = test::size(start, end).0;

            if skip < nrows {
                let hint = r.mut_iter().skip(skip).size_hint();

                let left = nrows - skip;

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
        if let Some(r) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_row(row)) {
            let nrows = test::size(start, end).0;
            let (start_row, start_col) = start;

            TestResult::from_bool(r.iter().rev().enumerate().all(|(col, e)| {
                e.eq(&(start_row + nrows - col - 1, start_col + row))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_mut_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        row: uint,
    ) -> TestResult {
        if let Some(mut r) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_row(row)) {
            let nrows = test::size(start, end).0;
            let (start_row, start_col) = start;

            TestResult::from_bool(r.mut_iter().rev().enumerate().all(|(col, e)| {
                e.eq(&&mut (start_row + nrows - col - 1, start_col + row))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, skip): (uint, uint),
    ) -> TestResult {
        if let Some(r) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_row(row)) {
            let nrows = test::size(start, end).0;

            if skip < nrows {
                let hint = r.iter().skip(skip).size_hint();

                let left = nrows - skip;

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }
}
