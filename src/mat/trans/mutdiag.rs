#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{Collection, Iter, MatrixMutDiag, MutIter, OptionIndex, OptionIndexMut, Transpose};

    #[quickcheck]
    fn at(size: (uint, uint), diag: int, index: uint) -> TestResult {
        if let Some(e) = test::mat(size).map(|m| m.t()).as_mut().and_then(|t| {
            t.mut_diag(diag)
        }).as_ref().and_then(|d| d.at(&index)) {
            let diag = -diag;

            let (row, col) = if diag > 0 {
                (index, index + diag as uint)
            } else {
                (index - diag as uint, index)
            };

            TestResult::from_bool((row, col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn at_mut(size: (uint, uint), diag: int, index: uint) -> TestResult {
        if let Some(e) = test::mat(size).map(|m| m.t()).as_mut().and_then(|m| {
            m.mut_diag(diag)
        }).as_mut().and_then(|d| d.at_mut(&index)) {
            let diag = -diag;

            let (row, col) = if diag > 0 {
                (index, index + diag as uint)
            } else {
                (index - diag as uint, index)
            };

            TestResult::from_bool((row, col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), diag: int) -> TestResult {
        if let Some(d) = test::mat(size).map(|m| m.t()).as_mut().and_then(|m| m.mut_diag(diag)) {
            let diag = -diag;

            if diag > 0 {
                TestResult::from_bool(d.iter().enumerate().all(|(i, e)| {
                    e.eq(&(i, i + diag as uint))
                }))
            } else {
                TestResult::from_bool(d.iter().enumerate().all(|(i, e)| {
                    e.eq(&(i - diag as uint, i))
                }))
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_iter(size: (uint, uint), diag: int) -> TestResult {
        if let Some(mut d) = test::mat(size).map(|m| m.t()).as_mut().and_then(|m| {
            m.mut_diag(diag)
        }) {
            let diag = -diag;

            if diag > 0 {
                TestResult::from_bool(d.mut_iter().enumerate().all(|(i, e)| {
                    e.eq(&&mut (i, i + diag as uint))
                }))
            } else {
                TestResult::from_bool(d.mut_iter().enumerate().all(|(i, e)| {
                    e.eq(&&mut (i - diag as uint, i))
                }))
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_size_hint(size: (uint, uint), diag: int, skip: uint) -> TestResult {
        if let Some(mut d) = test::mat(size).map(|m| m.t()).as_mut().and_then(|m| {
            m.mut_diag(diag)
        }) {
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
    fn rev_iter(size: (uint, uint), diag: int) -> TestResult {
        if let Some(d) = test::mat(size).map(|m| m.t()).as_mut().and_then(|m| m.mut_diag(diag)) {
            let diag = -diag;
            let n = d.len();

            if diag > 0 {
                TestResult::from_bool(d.iter().rev().enumerate().all(|(i, e)| {
                    e.eq(&(n - i - 1, n - i - 1 + diag as uint))
                }))
            } else {
                TestResult::from_bool(d.iter().rev().enumerate().all(|(i, e)| {
                    e.eq(&(n - i - 1 - diag as uint, n - i - 1))
                }))
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn rev_mut_iter(size: (uint, uint), diag: int) -> TestResult {
        if let Some(mut d) = test::mat(size).map(|m| m.t()).as_mut().and_then(|m| {
            m.mut_diag(diag)
        }) {
            let diag = -diag;
            let n = d.len();

            if diag > 0 {
                TestResult::from_bool(d.mut_iter().rev().enumerate().all(|(i, e)| {
                    e.eq(&&mut (n - i - 1, n - i - 1 + diag as uint))
                }))
            } else {
                TestResult::from_bool(d.mut_iter().rev().enumerate().all(|(i, e)| {
                    e.eq(&&mut (n - i - 1 - diag as uint, n - i - 1))
                }))
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), diag: int, skip: uint) -> TestResult {
        if let Some(d) = test::mat(size).map(|m| m.t()).as_mut().and_then(|m| m.mut_diag(diag)) {
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
