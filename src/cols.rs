use cast::From;
use extract::Extract;

use {Col, ColMut, Cols, ColsMut};

impl<'a, T> Clone for Cols<'a, T> {
    fn clone(&self) -> Cols<'a, T> {
        Cols {
            ..*self
        }
    }
}

impl<'a, T> DoubleEndedIterator for Cols<'a, T> {
    fn next_back(&mut self) -> Option<Col<'a, T>> {
        unsafe {
            if self.0.ncols == 0 {
                None
            } else {
                let col = self.0.unsafe_col(self.0.ncols - 1);
                self.0 = self.0.unsafe_slice((0, 0)..(self.0.nrows, self.0.ncols - 1));
                Some(col)
            }
        }
    }
}

impl<'a, T> Iterator for Cols<'a, T> {
    type Item = Col<'a, T>;

    fn next(&mut self) -> Option<Col<'a, T>> {
        unsafe {
            if self.0.ncols == 0 {
                None
            } else {
                let col = self.0.unsafe_col(0);
                self.0 = self.0.unsafe_slice((0, 1)..(self.0.nrows, self.0.ncols));
                Some(col)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        unsafe {
            let exact = usize::from(self.0.ncols).extract();
            (exact, Some(exact))
        }
    }
}

impl<'a, T> DoubleEndedIterator for ColsMut<'a, T> {
    fn next_back(&mut self) -> Option<ColMut<'a, T>> {
        self.0.next_back().map(ColMut)
    }
}

impl<'a, T> Iterator for ColsMut<'a, T> {
    type Item = ColMut<'a, T>;

    fn next(&mut self) -> Option<ColMut<'a, T>> {
        self.0.next().map(ColMut)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
