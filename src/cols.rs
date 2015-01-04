use std::mem;

use traits::{MatrixCol, MatrixColMut};
use {Col, Cols, MutCol, MutCols};

impl<'a, T, M> DoubleEndedIterator for Cols<'a, M> where M: MatrixCol<T> {
    fn next_back(&mut self) -> Option<Col<'a, T>> {
        self.0.next_back()
    }
}

impl<'a, T, M> Iterator for Cols<'a, M> where M: MatrixCol<T> {
    type Item = Col<'a, T>;
    fn next(&mut self) -> Option<Col<'a, T>> {
        self.0.next()
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.0.size_hint()
    }
}

impl<'a, T, M> DoubleEndedIterator for MutCols<'a, M> where M: MatrixColMut<T> {
    fn next_back(&mut self) -> Option<MutCol<'a, T>> {
        unsafe { mem::transmute(self.0.next_back()) }
    }
}

impl<'a, T, M> Iterator for MutCols<'a, M> where M: MatrixColMut<T> {
    type Item = MutCol<'a, T>;
    fn next(&mut self) -> Option<MutCol<'a, T>> {
        unsafe { mem::transmute(self.0.next()) }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.0.size_hint()
    }
}
