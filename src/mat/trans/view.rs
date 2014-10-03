#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{OptionIndex, OptionSlice, Transpose};

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        if let Some(e) = test::mat(size).map(|m| m.t()).as_ref().and_then(|t| {
            t.slice(start, end)
        }).as_ref().and_then(|v| v.at(&(row, col))) {
            let (start_row, start_col) = start;

            TestResult::from_bool((start_col + col, start_row + row).eq(e))
        } else {
            TestResult::discard()
        }
    }
}
