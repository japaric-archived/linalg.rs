use transposed::{Cols, ColsMut};
use {Col, ColMut};

impl<'a, T> DoubleEndedIterator for Cols<'a, T> {
    fn next_back(&mut self) -> Option<Col<'a, T>> {
        self.0.next_back().map(|r| Col(r.0))
    }
}

impl<'a, T> Iterator for Cols<'a, T> {
    type Item = Col<'a, T>;

    fn next(&mut self) -> Option<Col<'a, T>> {
        self.0.next().map(|r| Col(r.0))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for ColsMut<'a, T> {
    fn next_back(&mut self) -> Option<ColMut<'a, T>> {
        self.0.next_back().map(|r| ColMut(Col((r.0).0)))
    }
}

impl<'a, T> Iterator for ColsMut<'a, T> {
    type Item = ColMut<'a, T>;

    fn next(&mut self) -> Option<ColMut<'a, T>> {
        self.0.next().map(|r| ColMut(Col((r.0).0)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
