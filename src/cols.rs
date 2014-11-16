use traits::{MatrixCol, MatrixColMut};
use {Col, Cols, MutCols};

impl<'a, V, M> DoubleEndedIterator<Col<V>> for Cols<'a, M> where
    M: MatrixCol<'a, V>,
{
    fn next_back(&mut self) -> Option<Col<V>> {
        if self.state == self.stop {
            None
        } else {
            self.stop -= 1;
            Some(unsafe { self.mat.unsafe_col(self.stop) })
        }
    }
}

impl<'a, V, M> Iterator<Col<V>> for Cols<'a, M> where
    M: MatrixCol<'a, V>,
{
    fn next(&mut self) -> Option<Col<V>> {
        if self.state == self.stop {
            None
        } else {
            let col = unsafe { self.mat.unsafe_col(self.state) };
            self.state += 1;
            Some(col)
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let exact = self.stop - self.state;
        (exact, Some(exact))
    }
}

impl<'a, V, M> DoubleEndedIterator<Col<V>> for MutCols<'a, M> where
    M: MatrixColMut<'a, V>,
{
    fn next_back(&mut self) -> Option<Col<V>> {
        if self.state == self.stop {
            None
        } else {
            self.stop -= 1;
            // NB This doesn't *really* alias memory, because the columns are not overlapping
            Some(unsafe { (*(self.mat as *mut _)).unsafe_col_mut(self.stop) })
        }
    }
}

impl<'a, V, M> Iterator<Col<V>> for MutCols<'a, M> where M: MatrixColMut<'a, V> {
    fn next(&mut self) -> Option<Col<V>> {
        if self.state == self.stop {
            None
        } else {
            // NB This doesn't *really* alias memory, because the columns are not overlapping
            let col = unsafe { (*(self.mat as *mut _)).unsafe_col_mut(self.state) };
            self.state += 1;
            Some(col)
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let exact = self.stop - self.state;
        (exact, Some(exact))
    }
}
