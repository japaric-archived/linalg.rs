use transposed::{Rows, RowsMut};
use {Row, RowMut};

impl<'a, T> DoubleEndedIterator for Rows<'a, T> {
    fn next_back(&mut self) -> Option<Row<'a, T>> {
        self.0.next_back().map(|c| Row(c.0))
    }
}

impl<'a, T> Iterator for Rows<'a, T> {
    type Item = Row<'a, T>;

    fn next(&mut self) -> Option<Row<'a, T>> {
        self.0.next().map(|c| Row(c.0))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for RowsMut<'a, T> {
    fn next_back(&mut self) -> Option<RowMut<'a, T>> {
        self.0.next_back().map(|c| RowMut(Row((c.0).0)))
    }
}

impl<'a, T> Iterator for RowsMut<'a, T> {
    type Item = RowMut<'a, T>;

    fn next(&mut self) -> Option<RowMut<'a, T>> {
        self.0.next().map(|c| RowMut(Row((c.0).0)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
