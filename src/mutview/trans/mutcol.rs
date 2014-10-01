#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{
        Iter,
        MatrixMutCol,
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
        }).as_mut().and_then(|t| t.mut_col(col)).as_ref().and_then(|c| c.at(&row)) {
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
        }).as_mut().and_then(|t| t.mut_col(col)).as_mut().and_then(|c| c.at_mut(&row)) {
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
        col: uint,
    ) -> TestResult {
        if let Some(c) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_col(col)) {
            let (start_row, start_col) = start;

            TestResult::from_bool(c.iter().enumerate().all(|(row, e)| {
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
        col: uint,
    ) -> TestResult {
        if let Some(mut c) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_col(col)) {
            let (start_row, start_col) = start;

            TestResult::from_bool(c.mut_iter().enumerate().all(|(row, e)| {
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
        (col, skip): (uint, uint),
    ) -> TestResult {
        if let Some(mut c) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).map(|v| v.t()).as_mut().and_then(|t| t.mut_col(col)) {
            let (_, ncols) = test::size(start, end);

            if skip < ncols {
                let hint = c.mut_iter().skip(skip).size_hint();

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
        col: uint,
    ) -> TestResult {
        if let Some(c) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_col(col)) {
            let (_, ncols) = test::size(start, end);
            let (start_row, start_col) = start;

            TestResult::from_bool(c.iter().rev().enumerate().all(|(row, e)| {
                e.eq(&(start_row + col, start_col + ncols - row - 1))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_mut_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        col: uint,
    ) -> TestResult {
        if let Some(mut c) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).map(|v| v.t()).as_mut().and_then(|t| t.mut_col(col)) {
            let (_, ncols) = test::size(start, end);
            let (start_row, start_col) = start;

            TestResult::from_bool(c.mut_iter().rev().enumerate().all(|(row, e)| {
                e.eq(&&mut (start_row + col, start_col + ncols - row - 1))
            }))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (col, skip): (uint, uint),
    ) -> TestResult {
        if let Some(c) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_col(col)) {
            let (_, ncols) = test::size(start, end);

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
