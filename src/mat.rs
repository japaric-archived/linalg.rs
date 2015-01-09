use std::{cmp, mem};

use {Col, Diag, Error, Mat, MutCol, MutDiag, MutRow, Result, Row};
use traits::{
    Matrix, MatrixCol, MatrixCols, MatrixColMut, MatrixDiag, MatrixDiagMut, MatrixMutCols,
    MatrixMutRows, MatrixRow, MatrixRows, MatrixRowMut,
};

impl<T> Matrix for Mat<T> {
    type Elem = T;

    fn ncols(&self) -> usize { self.ncols }
    fn nrows(&self) -> usize { self.nrows }
}

impl<T> MatrixCol for Mat<T> {
    unsafe fn unsafe_col(&self, col: usize) -> Col<T> {
        Col(::From::parts((
            self.data.as_ptr().offset((col * self.nrows()) as isize),
            self.nrows(),
            1,
        )))
    }
}

impl<T> MatrixColMut for Mat<T> {
    unsafe fn unsafe_col_mut(&mut self, col: usize) -> MutCol<T> {
        mem::transmute(self.unsafe_col(col))
    }
}

impl<T> MatrixCols for Mat<T> {}

impl<T> MatrixDiag for Mat<T> {
    fn diag(&self, diag: isize) -> Result<Diag<T>> {
        let (nrows, ncols) = (self.nrows, self.ncols);
        let stride = nrows;

        if diag > 0 {
            let diag = diag as usize;

            if diag < ncols {
                let ptr = unsafe { self.data.as_ptr().offset((diag * stride) as isize) };
                let len = cmp::min(nrows, ncols - diag);

                Ok(Diag(unsafe { ::From::parts((ptr, len, stride + 1)) }))
            } else {
                Err(Error::NoSuchDiagonal)
            }
        } else {
            let diag = -diag as usize;

            if diag < nrows {
                let ptr = unsafe { self.data.as_ptr().offset(diag as isize) };
                let len = cmp::min(nrows - diag, ncols);

                Ok(Diag(unsafe { ::From::parts((ptr, len, stride + 1)) }))
            } else {
                Err(Error::NoSuchDiagonal)
            }
        }
    }
}

impl<T> MatrixDiagMut for Mat<T> {
    fn diag_mut(&mut self, diag: isize) -> Result<MutDiag<T>> {
        unsafe { mem::transmute(self.diag(diag)) }
    }
}

impl<T> MatrixMutCols for Mat<T> {}

impl<T> MatrixMutRows for Mat<T> {}

impl<T> MatrixRow for Mat<T> {
    unsafe fn unsafe_row(&self, row: usize) -> Row<T> {
        let (nrows, ncols) = self.size();

        Row(::From::parts((
            self.data.as_ptr().offset(row as isize),
            ncols,
            nrows,
        )))
    }
}

impl<T> MatrixRowMut for Mat<T> {
    unsafe fn unsafe_row_mut(&mut self, row: usize) -> MutRow<T> {
        mem::transmute(self.unsafe_row(row))
    }
}

impl<T> MatrixRows for Mat<T> {}
