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

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, T> Iterator for MutItems<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        unsafe { mem::transmute(self.0.next()) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, T> ::From<(*const T, usize, usize, usize)> for MutView<'a, T> {
    unsafe fn parts(parts: (*const T, usize, usize, usize)) -> MutView<'a, T>{
        MutView(::From::parts(parts))
    }
}

impl<'a, T> ::From<(*const T, usize, usize, usize)> for View<'a, T> {
    unsafe fn parts(parts: (*const T, usize, usize, usize)) -> View<'a, T>{
        View(::From::parts(parts))
    }
}

impl<'a, 'b, T> IterMut<'b> for MutView<'a, T> {
    type Iter = MutItems<'b, T>;

    fn iter_mut(&'b mut self) -> MutItems<'b, T> {
        MutItems(self.0.iter())
    }
}

impl<'a, T> MatrixColMut for MutView<'a, T> {
    unsafe fn unsafe_col_mut(&mut self, col: usize) -> MutCol<T> {
        mem::transmute(self.0.unsafe_col(col))
    }
}

impl<'a, T> MatrixDiagMut for MutView<'a, T> {
    fn diag_mut(&mut self, diag: isize) -> Result<MutDiag<T>> {
        unsafe { mem::transmute(self.0.diag(diag)) }
    }
}

impl<'a, T> MatrixMutCols for MutView<'a, T> {}

impl<'a, T> MatrixMutRows for MutView<'a, T> {}

impl<'a, T> MatrixRowMut for MutView<'a, T> {
    unsafe fn unsafe_row_mut(&mut self, row: usize) -> MutRow<T> {
        mem::transmute(self.0.unsafe_row(row))
    }
}

macro_rules! impls {
    ($($ty:ty),+) => {
        $(
            impl<'a, 'b, T> Iter<'b> for $ty {
                type Iter = Items<'b, T>;

                fn iter(&'b self) -> Items<'b, T> {
                    Items(self.0.iter())
                }
            }

            impl<'a, T> Matrix for $ty {
                type Elem = T;

                fn ncols(&self) -> usize {
                    self.0.ncols()
                }

                fn nrows(&self) -> usize {
                    self.0.nrows()
                }
            }

            impl<'a, T> MatrixCol for $ty {
                unsafe fn unsafe_col(&self, col: usize) -> Col<T> {
                    self.0.unsafe_col(col)
                }
            }

            impl<'a, T> MatrixCols for $ty {}

            impl<'a, T> MatrixDiag for $ty {
                fn diag(&self, diag: isize) -> Result<Diag<T>> {
                    self.0.diag(diag)
                }
            }

            impl<'a, T> MatrixRow for $ty {
                unsafe fn unsafe_row(&self, row: usize) -> Row<T> {
                    self.0.unsafe_row(row)
                }
            }

            impl<'a, T> MatrixRows for $ty {}
        )+
    }
}

impls!(MutView<'a, T>, View<'a, T>);
