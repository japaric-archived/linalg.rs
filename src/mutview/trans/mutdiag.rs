#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{
        Iter,
        MatrixMutDiag,
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
        (diag, index): (int, uint),
    ) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_diag(diag)).as_ref().and_then(|d| d.at(&index)) {
            let diag = -diag;
            let (start_row, start_col) = start;

            let (row, col) = if diag > 0 {
                (start_row + index, start_col + index + diag as uint)
            } else {
                (start_row + index - diag as uint, start_col + index)
            };

            TestResult::from_bool((row, col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn at_mut(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (diag, index): (int, uint),
    ) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_diag(diag)).as_mut().and_then(|d| d.at_mut(&index)) {
            let diag = -diag;
            let (start_row, start_col) = start;

            let (row, col) = if diag > 0 {
                (start_row + index, start_col + index + diag as uint)
            } else {
                (start_row + index - diag as uint, start_col + index)
            };

            TestResult::from_bool((row, col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        diag: int,
    ) -> TestResult {
        if let Some(d) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_diag(diag)) {
            let diag = -diag;
            let (start_row, start_col) = start;

            if diag > 0 {
                TestResult::from_bool(d.iter().enumerate().all(|(i, e)| {
                    e.eq(&(start_row + i, start_col + i + diag as uint))
                }))
            } else {
                TestResult::from_bool(d.iter().enumerate().all(|(i, e)| {
                    e.eq(&(start_row + i - diag as uint, start_col + i))
                }))
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        diag: int,
    ) -> TestResult {
        if let Some(mut d) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).map(|v| v.t()).as_mut().and_then(|t| t.mut_diag(diag)) {
            let diag = -diag;
            let (start_row, start_col) = start;

            if diag > 0 {
                TestResult::from_bool(d.mut_iter().enumerate().all(|(i, e)| {
                    e.eq(&&mut (start_row + i, start_col + i + diag as uint))
                }))
            } else {
                TestResult::from_bool(d.mut_iter().enumerate().all(|(i, e)| {
                    e.eq(&&mut (start_row + i - diag as uint, start_col + i))
                }))
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (diag, skip): (int, uint),
    ) -> TestResult {
        if let Some(mut d) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).map(|v| v.t()).as_mut().and_then(|t| t.mut_diag(diag)) {
            let n = d.len();

            if skip < n {
                let hint = d.mut_iter().skip(skip).size_hint();

                let left = n - skip;

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
        diag: int,
    ) -> TestResult {
        if let Some(d) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_diag(diag)) {
            let diag = -diag;
            let (start_row, start_col) = start;
            let n = d.len();

            if diag > 0 {
                TestResult::from_bool(d.iter().rev().enumerate().all(|(i, e)| {
                    e.eq(&(start_row + n - i - 1, start_col + n - i - 1 + diag as uint))
                }))
            } else {
                TestResult::from_bool(d.iter().rev().enumerate().all(|(i, e)| {
                    e.eq(&(start_row + n - i - 1 - diag as uint, start_col + n - i - 1))
                }))
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_mut_iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        diag: int,
    ) -> TestResult {
        if let Some(mut d) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).map(|v| v.t()).as_mut().and_then(|t| t.mut_diag(diag)) {
            let diag = -diag;
            let (start_row, start_col) = start;
            let n = d.len();

            if diag > 0 {
                TestResult::from_bool(d.mut_iter().rev().enumerate().all(|(i, e)| {
                    e.eq(&&mut (start_row + n - i - 1, start_col + n - i - 1 + diag as uint))
                }))
            } else {
                TestResult::from_bool(d.mut_iter().rev().enumerate().all(|(i, e)| {
                    e.eq(&&mut (start_row + n - i - 1 - diag as uint, start_col + n - i - 1))
                }))
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (diag, skip): (int, uint),
    ) -> TestResult {
        if let Some(d) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.mut_diag(diag)) {
            let n = d.len();

            if skip < n {
                let hint = d.iter().skip(skip).size_hint();

                let left = n - skip;

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }
}
