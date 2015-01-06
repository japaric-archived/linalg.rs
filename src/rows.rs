use std::mem;

use traits::{MatrixRow, MatrixRowMut};
use {Row, Rows, MutRow, MutRows};

impl<'a, T, M> DoubleEndedIterator for Rows<'a, M> where M: MatrixRow<T> {
    fn next_back(&mut self) -> Option<Row<'a, T>> {
        self.0.next_back()
    }
}

impl<'a, T, M> Iterator for Rows<'a, M> where M: MatrixRow<T> {
    type Item = Row<'a, T>;

    fn next(&mut self) -> Option<Row<'a, T>> {
        self.0.next()
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.0.size_hint()
    }
}

impl<'a, T, M> DoubleEndedIterator for MutRows<'a, M> where M: MatrixRowMut<T> {
    fn next_back(&mut self) -> Option<MutRow<'a, T>> {
        unsafe { mem::transmute(self.0.next_back()) }
    }
}

impl<'a, T, M> Iterator for MutRows<'a, M> where M: MatrixRowMut<T> {
    type Item = MutRow<'a, T>;

    fn next(&mut self) -> Option<MutRow<'a, T>> {
        unsafe { mem::transmute(self.0.next()) }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.0.size_hint()
    }
}
