use Col;
use traits::{Matrix, MatrixCol};

pub struct Cols<'a, M: 'a> {
    mat: &'a M,
    state: usize,
    stop: usize,
}

impl<'a, M> Copy for Cols<'a, M> {}

impl<'a, T, M> DoubleEndedIterator for Cols<'a, M> where M: MatrixCol + Matrix<Elem=T> {
    fn next_back(&mut self) -> Option<Col<'a, T>> {
        if self.state == self.stop {
            None
        } else {
            self.stop -= 1;
            Some(unsafe { self.mat.unsafe_col(self.stop) })
        }
    }
}

impl<'a, M> ::From<&'a M> for Cols<'a, M> where M: Matrix {
    unsafe fn parts(mat: &'a M) -> Cols<'a, M> {
        Cols {
            mat: mat,
            state: 0,
            stop: mat.ncols(),
        }
    }
}

impl<'a, T, M> Iterator for Cols<'a, M> where M: MatrixCol + Matrix<Elem=T> {
    type Item = Col<'a, T>;

    fn next(&mut self) -> Option<Col<'a, T>> {
        if self.state == self.stop {
            None
        } else {
            let col = unsafe { self.mat.unsafe_col(self.state) };
            self.state += 1;
            Some(col)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.stop - self.state;
        (exact, Some(exact))
    }
}
