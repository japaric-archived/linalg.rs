use cast::From;
use extract::Extract;

use {Row, RowMut, Rows, RowsMut};

impl<'a, T> Clone for Rows<'a, T> {
    fn clone(&self) -> Rows<'a, T> {
        Rows {
            ..*self
        }
    }
}

impl<'a, T> DoubleEndedIterator for Rows<'a, T> {
    fn next_back(&mut self) -> Option<Row<'a, T>> {
        unsafe {
            if self.0.nrows == 0 {
                None
            } else {
                let row = self.0.unsafe_row(self.0.nrows - 1);
                self.0 = self.0.unsafe_slice((0, 0)..(self.0.nrows - 1, self.0.ncols));
                Some(row)
            }
        }
    }
}

impl<'a, T> Iterator for Rows<'a, T> {
    type Item = Row<'a, T>;

    fn next(&mut self) -> Option<Row<'a, T>> {
        unsafe {
            if self.0.nrows == 0 {
                None
            } else {
                let row = self.0.unsafe_row(0);
                self.0 = self.0.unsafe_slice((1, 0)..(self.0.nrows, self.0.ncols));
                Some(row)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        unsafe {
            let exact = usize::from(self.0.nrows).extract();
            (exact, Some(exact))
        }
    }
}


impl<'a, T> DoubleEndedIterator for RowsMut<'a, T> {
    fn next_back(&mut self) -> Option<RowMut<'a, T>> {
        self.0.next_back().map(RowMut)
    }
}

impl<'a, T> Iterator for RowsMut<'a, T> {
    type Item = RowMut<'a, T>;

    fn next(&mut self) -> Option<RowMut<'a, T>> {
        self.0.next().map(RowMut)

    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
