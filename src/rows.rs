use traits::{MatrixRow, MatrixRowMut};
use {MutRows, Row, Rows};

impl<'a, V, M> DoubleEndedIterator<Row<V>> for Rows<'a, M> where
    M: MatrixRow<'a, V>,
{
    fn next_back(&mut self) -> Option<Row<V>> {
        if self.state == self.stop {
            None
        } else {
            self.stop -= 1;
            Some(unsafe { self.mat.unsafe_row(self.stop) })
        }
    }
}

impl<'a, V, M> Iterator<Row<V>> for Rows<'a, M> where
    M: MatrixRow<'a, V>,
{
    fn next(&mut self) -> Option<Row<V>> {
        if self.state == self.stop {
            None
        } else {
            let row = unsafe { self.mat.unsafe_row(self.state) };
            self.state += 1;
            Some(row)
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let exact = self.stop - self.state;
        (exact, Some(exact))
    }
}

impl<'a, V, M> DoubleEndedIterator<Row<V>> for MutRows<'a, M> where
    M: MatrixRowMut<'a, V>,
{
    fn next_back(&mut self) -> Option<Row<V>> {
        if self.state == self.stop {
            None
        } else {
            self.stop -= 1;
            // NB This doesn't *really* alias memory, because the rows are not overlapping
            Some(unsafe { (*(self.mat as *mut _)).unsafe_row_mut(self.stop) })
        }
    }
}

impl<'a, V, M> Iterator<Row<V>> for MutRows<'a, M> where M: MatrixRowMut<'a, V> {
    fn next(&mut self) -> Option<Row<V>> {
        if self.state == self.stop {
            None
        } else {
            // NB This doesn't *really* alias memory, because the rows are not overlapping
            let row = unsafe { (*(self.mat as *mut _)).unsafe_row_mut(self.state) };
            self.state += 1;
            Some(row)
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let exact = self.stop - self.state;
        (exact, Some(exact))
    }
}
