use std::{cmp, mem, raw};

use traits::{
    Matrix, MatrixCol, MatrixColMut, MatrixCols, MatrixDiag, MatrixDiagMut, MatrixMutCols,
    MatrixMutRows, MatrixRow, MatrixRowMut, MatrixRows, Transpose,
};
use {Col, Diag, Mat, Row, Trans, strided};

impl<T> Matrix for Mat<T> {
    fn size(&self) -> (uint, uint) {
        self.size
    }
}

// FIXME (DRY) Merge these two impls using a macro
impl<'a, T> MatrixCol<'a, &'a [T]> for Mat<T> {
    unsafe fn unsafe_col(&self, col: uint) -> Col<&[T]> {
        Col(mem::transmute(raw::Slice {
            data: self.data.as_ptr().offset((col * self.nrows()) as int),
            len: self.nrows(),
        }))
    }
}

impl<'a, T> MatrixColMut<'a, &'a mut [T]> for Mat<T> {
    unsafe fn unsafe_col_mut(&mut self, col: uint) -> Col<&mut [T]> {
        Col(mem::transmute(raw::Slice {
            data: self.data.as_ptr().offset((col * self.nrows()) as int),
            len: self.nrows(),
        }))
    }
}

impl<'a, T> MatrixCols<'a> for Mat<T> {}

// FIXME (DRY) Merge these two impls using a macro
impl<T> MatrixDiag<T> for Mat<T> {
    fn diag(&self, diag: int) -> ::Result<Diag<strided::Slice<T>>> {
        let (nrows, ncols) = self.size;
        let stride = nrows;

        if diag > 0 {
            let diag = diag as uint;

            if diag < ncols {
                let ptr = unsafe { self.data.as_ptr().offset((diag * stride) as int) };
                let len = cmp::min(nrows, ncols - diag);

                Ok(Diag(unsafe { ::Strided::from_parts(ptr, len, stride + 1) }))
            } else {
                Err(::NoSuchDiagonal)
            }
        } else {
            let diag = -diag as uint;

            if diag < nrows {
                let ptr = unsafe { self.data.as_ptr().offset(diag as int) };
                let len = cmp::min(nrows - diag, ncols);

                Ok(Diag(unsafe { ::Strided::from_parts(ptr, len, stride + 1) }))
            } else {
                Err(::NoSuchDiagonal)
            }
        }
    }
}

impl<T> MatrixDiagMut<T> for Mat<T> {
    fn diag_mut(&mut self, diag: int) -> ::Result<Diag<strided::MutSlice<T>>> {
        let (nrows, ncols) = self.size;
        let stride = nrows;

        if diag > 0 {
            let diag = diag as uint;

            if diag < ncols {
                let ptr = unsafe { self.data.as_ptr().offset((diag * stride) as int) };
                let len = cmp::min(nrows, ncols - diag);

                Ok(Diag(unsafe { ::Strided::from_parts(ptr, len, stride + 1) }))
            } else {
                Err(::NoSuchDiagonal)
            }
        } else {
            let diag = -diag as uint;

            if diag < nrows {
                let ptr = unsafe { self.data.as_ptr().offset(diag as int) };
                let len = cmp::min(nrows - diag, ncols);

                Ok(Diag(unsafe { ::Strided::from_parts(ptr, len, stride + 1) }))
            } else {
                Err(::NoSuchDiagonal)
            }
        }
    }
}

impl<'a, T> MatrixMutCols<'a> for Mat<T> {}

impl<'a, T> MatrixMutRows<'a> for Mat<T> {}

// FIXME (DRY) Merge these two impls using a macro
impl<'a, T> MatrixRow<'a, strided::Slice<'a, T>> for Mat<T> {
    unsafe fn unsafe_row(&'a self, row: uint) -> Row<strided::Slice<'a, T>> {
        let (nrows, ncols) = self.size();

        Row(::Strided::from_parts(
            self.data.as_ptr().offset(row as int),
            ncols,
            nrows,
        ))
    }
}

impl<'a, T> MatrixRowMut<'a, strided::MutSlice<'a, T>> for Mat<T> {
    unsafe fn unsafe_row_mut(&'a mut self, row: uint) -> Row<strided::MutSlice<'a, T>> {
        let (nrows, ncols) = self.size();

        Row(::Strided::from_parts(
            self.data.as_ptr().offset(row as int),
            ncols,
            nrows,
        ))
    }
}

impl<'a, T> MatrixRows<'a> for Mat<T> {}

impl<T> Transpose<Trans<Mat<T>>> for Mat<T> {
    fn t(self) -> Trans<Mat<T>> {
        Trans(self)
    }
}
