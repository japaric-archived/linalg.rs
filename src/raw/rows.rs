use Row;
use traits::{Matrix, MatrixRow};

pub struct Rows<'a, M: 'a> {
    mat: &'a M,
    state: uint,
    stop: uint,
}

impl<'a, M> Copy for Rows<'a, M> {}

impl<'a, T, M> DoubleEndedIterator<Row<'a, T>> for Rows<'a, M> where M: MatrixRow<T> {
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

impl<'a, T, M> Iterator<Row<'a, T>> for Rows<'a, M> where M: MatrixRow<T> {
    fn next(&mut self) -> Option<Row<'a, T>> {
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
