use std::{cmp, mem};

use {Col, Diag, Error, Mat, MutCol, MutDiag, MutRow, Result, Row};
use traits::{
    Matrix, MatrixCol, MatrixCols, MatrixColMut, MatrixDiag, MatrixDiagMut, MatrixMutCols,
    MatrixMutRows, MatrixRow, MatrixRows, MatrixRowMut,
};

impl<T> Matrix for Mat<T> {
    fn ncols(&self) -> uint { self.ncols }
    fn nrows(&self) -> uint { self.nrows }
}

impl<T> MatrixCol<T> for Mat<T> {
    unsafe fn unsafe_col(&self, col: uint) -> Col<T> {
        Col(::From::parts((
            self.data.as_ptr().offset((col * self.nrows()) as int),
            self.nrows(),
            1,
        )))
    }
}

impl<T> MatrixColMut<T> for Mat<T> {
    unsafe fn unsafe_col_mut(&mut self, col: uint) -> MutCol<T> {
        mem::transmute(self.unsafe_col(col))
    }
}

impl<T> MatrixCols for Mat<T> {}

impl<T> MatrixDiag<T> for Mat<T> {
    fn diag(&self, diag: int) -> Result<Diag<T>> {
        let (nrows, ncols) = (self.nrows, self.ncols);
        let stride = nrows;

        if diag > 0 {
            let diag = diag as uint;

            if diag < ncols {
                let ptr = unsafe { self.data.as_ptr().offset((diag * stride) as int) };
                let len = cmp::min(nrows, ncols - diag);

                Ok(Diag(unsafe { ::From::parts((ptr, len, stride + 1)) }))
            } else {
                Err(Error::NoSuchDiagonal)
            }
        } else {
            let diag = -diag as uint;

            if diag < nrows {
                let ptr = unsafe { self.data.as_ptr().offset(diag as int) };
                let len = cmp::min(nrows - diag, ncols);

                Ok(Diag(unsafe { ::From::parts((ptr, len, stride + 1)) }))
            } else {
                Err(Error::NoSuchDiagonal)
            }
        }
    }
}

impl<T> MatrixDiagMut<T> for Mat<T> {
    fn diag_mut(&mut self, diag: int) -> Result<MutDiag<T>> {
        unsafe { mem::transmute(self.diag(diag)) }
    }
}

impl<T> MatrixMutCols for Mat<T> {}

impl<T> MatrixMutRows for Mat<T> {}

impl<T> MatrixRow<T> for Mat<T> {
    unsafe fn unsafe_row(&self, row: uint) -> Row<T> {
        let (nrows, ncols) = self.size();

        Row(::From::parts((
            self.data.as_ptr().offset(row as int),
            ncols,
            nrows,
        )))
    }
}

impl<T> MatrixRowMut<T> for Mat<T> {
    unsafe fn unsafe_row_mut(&mut self, row: uint) -> MutRow<T> {
        mem::transmute(self.unsafe_row(row))
    }
}

impl<T> MatrixRows for Mat<T> {}
