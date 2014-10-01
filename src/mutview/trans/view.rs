#[cfg(test)]
mod test {
    use quickcheck::TestResult;

    use test;
    use traits::{OptionMutSlice, OptionSlice, OptionIndex, Transpose};

    #[quickcheck]
    fn at(
        (size, (row, col)): ((uint, uint), (uint, uint)),
        (start, end): ((uint, uint), (uint, uint)),
        (sub_start, sub_end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|m| {
            m.t()
        }).as_ref().and_then(|t| {
            t.slice(sub_start, sub_end)
        }).as_ref().and_then(|v| v.at(&(row, col))) {
            let (start_row, start_col) = start;
            let (sub_start_row, sub_start_col) = sub_start;
            let col_ = start_col + sub_start_row + row;
            let row_ = start_row + sub_start_col + col;

            TestResult::from_bool((row_, col_).eq(e))
        } else {
            TestResult::discard()
        }
    }
}
