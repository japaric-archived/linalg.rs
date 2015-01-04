use std::mem;

use {Col, Diag, Items, MutCol, MutDiag, MutItems, MutRow, MutView, Result, Row, View};
use traits::{
    Iter, IterMut, Matrix, MatrixCol, MatrixColMut, MatrixCols, MatrixDiag, MatrixDiagMut,
    MatrixMutCols, MatrixMutRows, MatrixRow, MatrixRowMut, MatrixRows,
};

impl<'a, T> Iterator for Items<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        self.0.next()
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.0.size_hint()
    }
}

impl<'a, T> Iterator for MutItems<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        unsafe { mem::transmute(self.0.next()) }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.0.size_hint()
    }
}

impl<'a, T> ::From<(*const T, uint, uint, uint)> for MutView<'a, T> {
    unsafe fn parts(parts: (*const T, uint, uint, uint)) -> MutView<'a, T>{
        MutView(::From::parts(parts))
    }
}

impl<'a, T> ::From<(*const T, uint, uint, uint)> for View<'a, T> {
    unsafe fn parts(parts: (*const T, uint, uint, uint)) -> View<'a, T>{
        View(::From::parts(parts))
    }
}

impl<'a, 'b, T> IterMut<'b, &'b mut T, MutItems<'b, T>> for MutView<'a, T> {
    fn iter_mut(&'b mut self) -> MutItems<'b, T> {
        MutItems(self.0.iter())
    }
}

impl<'a, T> MatrixColMut<T> for MutView<'a, T> {
    unsafe fn unsafe_col_mut(&mut self, col: uint) -> MutCol<T> {
        mem::transmute(self.0.unsafe_col(col))
    }
}

impl<'a, T> MatrixDiagMut<T> for MutView<'a, T> {
    fn diag_mut(&mut self, diag: int) -> Result<MutDiag<T>> {
        unsafe { mem::transmute(self.0.diag(diag)) }
    }
}

impl<'a, T> MatrixMutCols for MutView<'a, T> {}

impl<'a, T> MatrixMutRows for MutView<'a, T> {}

impl<'a, T> MatrixRowMut<T> for MutView<'a, T> {
    unsafe fn unsafe_row_mut(&mut self, row: uint) -> MutRow<T> {
        mem::transmute(self.0.unsafe_row(row))
    }
}

macro_rules! impls {
    ($($ty:ty),+) => {
        $(
            impl<'a, 'b, T> Iter<'b, &'b T, Items<'b, T>> for $ty {
                fn iter(&'b self) -> Items<'b, T> {
                    Items(self.0.iter())
                }
            }

            impl<'a, T> Matrix for $ty {
                fn ncols(&self) -> uint {
                    self.0.ncols()
                }

                fn nrows(&self) -> uint {
                    self.0.nrows()
                }
            }

            impl<'a, T> MatrixCol<T> for $ty {
                unsafe fn unsafe_col(&self, col: uint) -> Col<T> {
                    self.0.unsafe_col(col)
                }
            }

            impl<'a, T> MatrixCols for $ty {}

            impl<'a, T> MatrixDiag<T> for $ty {
                fn diag(&self, diag: int) -> Result<Diag<T>> {
                    self.0.diag(diag)
                }
            }

            impl<'a, T> MatrixRow<T> for $ty {
                unsafe fn unsafe_row(&self, row: uint) -> Row<T> {
                    self.0.unsafe_row(row)
                }
            }

            impl<'a, T> MatrixRows for $ty {}
        )+
    }
}

impls!(MutView<'a, T>, View<'a, T>);
