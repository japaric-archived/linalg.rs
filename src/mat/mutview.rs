use std::kinds::marker;

use traits::{Matrix, OptionMutSlice};
use {Mat, MutView};

impl<'a, T> OptionMutSlice<'a, (uint, uint), MutView<'a, T>> for Mat<T> {
    fn mut_slice(&'a mut self, start: (uint, uint), end: (uint, uint)) -> Option<MutView<'a, T>> {
        let (end_row, end_col) = end;
        let (nrows, ncols) = self.size();
        let (start_row, start_col) = start;

        if end_col <= ncols && end_col > start_col + 1 &&
                end_row <= nrows && end_row > start_row + 1 {
            let stride = self.stride;
            let ptr = unsafe {
                self.data.as_mut_ptr().offset((start_col * stride + start_row) as int)
            };

            Some(MutView {
                _contravariant: marker::ContravariantLifetime::<'a>,
                _nocopy: marker::NoCopy,
                _nosend: marker::NoSend,
                data: ptr,
                size: (end_row - start_row, end_col - start_col),
                stride: stride,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;
    use std::collections::TreeSet;

    use test;
    use traits::{Collection, Iter, MutIter, OptionMutSlice, OptionIndex, OptionIndexMut};

    #[quickcheck]
    fn at(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint),
    ) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_ref().and_then(|v| v.at(&(row, col))) {
            let (start_row, start_col) = start;

            TestResult::from_bool((start_row + row, start_col + col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn at_mut(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        (row, col): (uint, uint)
    ) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_mut().and_then(|v| v.at_mut(&(row, col))) {
            let (start_row, start_col) = start;

            TestResult::from_bool((start_row + row, start_col + col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint), (start, end): ((uint, uint), (uint, uint))) -> TestResult {
        if let Some(v) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)) {
            let (nrows, ncols) = test::size(start, end);
            let (start_row, start_col) = start;

            let mut elems = TreeSet::new();
            for row in range(0, nrows) {
                for col in range(0, ncols) {
                    elems.insert((start_row + row, start_col + col));
                }
            }

            TestResult::from_bool(elems == v.iter().map(|&x| x).collect())
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_iter(size: (uint, uint), (start, end): ((uint, uint), (uint, uint))) -> TestResult {
        if let Some(mut v) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)) {
            let (nrows, ncols) = test::size(start, end);
            let (start_row, start_col) = start;

            let mut elems = TreeSet::new();
            for row in range(0, nrows) {
                for col in range(0, ncols) {
                    elems.insert((start_row + row, start_col + col));
                }
            }

            TestResult::from_bool(elems == v.mut_iter().map(|&x| x).collect())
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_size_hint(
        size: (uint, uint),
        (start, end): ((uint, uint), (uint, uint)),
        skip: uint,
    ) -> TestResult {
        if let Some(mut v) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)) {
            let total = v.len();

            if skip < total {
                let left = total - skip;
                let hint = v.mut_iter().skip(skip).size_hint();

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
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
        if let Some(v) = test::mat(size).as_mut().and_then(|m| m.mut_slice(start, end)) {
            let total = v.len();

            if skip < total {
                let left = total - skip;
                let hint = v.iter().skip(skip).size_hint();

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }
}
