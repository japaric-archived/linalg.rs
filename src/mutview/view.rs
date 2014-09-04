use std::kinds::marker;

use {MutView, View};
use traits::{Matrix, OptionSlice};

impl<'a, 'b, T> OptionSlice<'b, (uint, uint), View<'b, T>> for MutView<'a, T> {
    fn slice(&'b self, start: (uint, uint), end: (uint, uint)) -> Option<View<'b, T>> {
        let (end_row, end_col) = end;
        let (nrows, ncols) = self.size();
        let (start_row, start_col) = start;

        if end_col < ncols && end_col > start_col + 1 &&
                end_row < nrows && end_row > start_row + 1 {
            let stride = self.stride;
            let ptr = unsafe {
                self.data.offset((start_row * stride + start_col) as int) as *const T
            };

            Some(View {
                _contravariant: marker::ContravariantLifetime::<'a>,
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

    use test;
    use traits::{OptionMutSlice, OptionIndex, OptionSlice};

    #[quickcheck]
    fn at(
        (size, (row, col)): ((uint, uint), (uint, uint)),
        (start, end): ((uint, uint), (uint, uint)),
        (sub_start, sub_end): ((uint, uint), (uint, uint)),
    ) -> TestResult {
        match test::mat(size).as_mut().and_then(|m| {
            m.mut_slice(start, end)
        }).as_ref().and_then(|v| {
            v.slice(sub_start, sub_end)
        }).as_ref().and_then(|v| v.at(&(row, col))) {
            None => TestResult::discard(),
            Some(e) => {
                let (start_row, start_col) = start;
                let (sub_start_row, sub_start_col) = sub_start;
                let col_ = sub_start_col + start_col + col;
                let row_ = sub_start_row + start_row + row;

                TestResult::from_bool((row_, col_).eq(e))
            },
        }
    }
}
