use Row;
use traits::{Matrix, MatrixRow};

pub struct Rows<'a, M> where M: 'a {
    mat: &'a M,
    state: usize,
    stop: usize,
}

impl<'a, M> Copy for Rows<'a, M> {}

impl<'a, M> Clone for Rows<'a, M> {
    fn clone(&self) -> Rows<'a, M> {
        *self
    }
}

impl<'a, T, M> DoubleEndedIterator for Rows<'a, M> where M: MatrixRow + Matrix<Elem=T> {
    fn next_back(&mut self) -> Option<Row<'a, T>> {
        if self.state == self.stop {
            None
        } else {
            self.stop -= 1;
            Some(unsafe { self.mat.unsafe_row(self.stop) })
        }
    }
}

impl<'a, M> ::From<&'a M> for Rows<'a, M> where M: Matrix {
    unsafe fn parts(mat: &'a M) -> Rows<'a, M> {
        Rows {
            mat: mat,
            state: 0,
            stop: mat.nrows(),
        }
    }
}

impl<'a, T, M> Iterator for Rows<'a, M> where M: MatrixRow + Matrix<Elem=T> {
    type Item = Row<'a, T>;

    fn next(&mut self) -> Option<Row<'a, T>> {
        if self.state == self.stop {
            None
        } else {
            let row = unsafe { self.mat.unsafe_row(self.state) };
            self.state += 1;
            Some(row)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.stop - self.state;
        (exact, Some(exact))
    }
}
