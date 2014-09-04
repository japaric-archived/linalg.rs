use traits::Transpose;
use {MutView, Trans};

mod col;
mod cols;
mod diag;
mod mutcol;
mod mutcols;
mod mutdiag;
mod mutrow;
mod mutrows;
mod mutview;
mod row;
mod rows;
mod view;

impl<'a, T> Transpose<Trans<MutView<'a, T>>> for MutView<'a, T> {
    fn t(self) -> Trans<MutView<'a, T>> {
        Trans {
            mat: self,
        }
    }
}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;
    use std::collections::TreeSet;

    use test;
    use traits::{Iter, MutIter, OptionMutSlice, OptionIndex, OptionIndexMut, Transpose};

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_ref().and_then(|t| t.at(&(row, col))) {
            None => TestResult::discard(),
            Some(e) => {
                let (start_row, start_col) = start;

                TestResult::from_bool((start_row + col, start_col + row).eq(e))
            },
        }
    }

    #[quickcheck]
    fn at_mut(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| {
            v.t()
        }).as_mut().and_then(|t| t.at_mut(&(row, col))) {
            None => TestResult::discard(),
            Some(e) => {
                let (start_row, start_col) = start;

                TestResult::from_bool((start_row + col, start_col + row).eq(e))
            },
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), (start, end): ((uint, uint), (uint, uint))) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| v.t()) {
            None => TestResult::discard(),
            Some(t) => {
                let (nrows, ncols) = test::size(start, end);
                let (start_row, start_col) = start;

                let mut elems = TreeSet::new();
                for row in range(0, nrows) {
                    for col in range(0, ncols) {
                        elems.insert((start_row + row, start_col + col));
                    }
                }

                TestResult::from_bool(elems == t.iter().map(|&x| x).collect())
            }
        }
    }

    #[quickcheck]
    fn mut_iter(size: (uint, uint), (start, end): ((uint, uint), (uint, uint))) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| v.t()) {
            None => TestResult::discard(),
            Some(mut t) => {
                let (nrows, ncols) = test::size(start, end);
                let (start_row, start_col) = start;

                let mut elems = TreeSet::new();
                for row in range(0, nrows) {
                    for col in range(0, ncols) {
                        elems.insert((start_row + row, start_col + col));
                    }
                }

                TestResult::from_bool(elems == t.mut_iter().map(|&x| x).collect())
            }
        }
    }

    #[quickcheck]
    fn mut_size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        skip: uint,
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| v.t()) {
            None => TestResult::discard(),
            Some(mut t) => {
                let total = t.len();

                if skip < total {
                    let hint = t.mut_iter().skip(skip).size_hint();
                    let left = total - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }

    #[quickcheck]
    fn size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        skip: uint,
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)).map(|v| v.t()) {
            None => TestResult::discard(),
            Some(t) => {
                let total = t.len();

                if skip < total {
                    let hint = t.iter().skip(skip).size_hint();
                    let left = total - skip;

                    TestResult::from_bool(hint == (left, Some(left)))
                } else {
                    TestResult::discard()
                }
            },
        }
    }
}
