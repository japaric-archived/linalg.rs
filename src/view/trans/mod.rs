use traits::Transpose;
use {Trans, View};

mod col;
mod cols;
mod diag;
mod row;
mod rows;
mod view;

impl<'a, T> Transpose<Trans<View<'a, T>>> for View<'a, T> {
    fn t(self) -> Trans<View<'a, T>> {
        Trans(self)
    }
}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;
    use std::collections::TreeSet;

    use test;
    use traits::{Iter, OptionSlice, OptionIndex, Transpose};

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        if let Some(e) = test::mat(size).as_ref().and_then(|m| m.slice(start, end)).map(|v| {
            v.t()
        }).as_ref().and_then(|t| t.at(&(row, col))) {
            let (start_row, start_col) = start;
            let row_ = start_row + col;
            let col_ = start_col + row;

            TestResult::from_bool((row_, col_).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        if let Some(t) = test::mat(size).as_ref().and_then(|m| {
            m.slice(start, end)
        }).map(|v| v.t()) {
            let (nrows, ncols) = test::size(start, end);
            let (start_row, start_col) = start;

            let mut elems = TreeSet::new();
            for row in range(0, nrows) {
                for col in range(0, ncols) {
                    elems.insert((start_row + row, start_col + col));
                }
            }

            TestResult::from_bool(elems == t.iter().map(|&x| x).collect())
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        skip: uint,
    ) -> TestResult {
        if let Some(t) = test::mat(size).as_ref().and_then(|m| {
            m.slice(start, end)
        }).map(|v| v.t()) {
            let total = t.len();

            if skip < total {
                let left = total - skip;
                let hint = t.iter().skip(skip).size_hint();

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }
}
