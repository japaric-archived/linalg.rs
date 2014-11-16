use {Col, Diag, Row, Trans, strided};
use traits::{
    At, AtMut, Iter, IterMut, Matrix, MatrixCol, MatrixColMut, MatrixCols, MatrixDiag,
    MatrixDiagMut, MatrixMutCols, MatrixMutRows, MatrixRow, MatrixRowMut, MatrixRows, Slice,
    SliceMut, Transpose,
};

impl<T, M> At<(uint, uint), T> for Trans<M> where M: At<(uint, uint), T> {
    fn at(&self, (row, col): (uint, uint)) -> ::Result<&T> {
        self.0.at((col, row))
    }
}

impl<T, M> AtMut<(uint, uint), T> for Trans<M> where M: AtMut<(uint, uint), T> {
    fn at_mut(&mut self, (row, col): (uint, uint)) -> ::Result<&mut T> {
        self.0.at_mut((col, row))
    }
}

impl<'a, T, I, M> Iter<'a, T, I> for Trans<M> where
    I: Iterator<T>,
    M: Iter<'a, T, I>,
{
    fn iter(&'a self) -> I {
        self.0.iter()
    }
}

impl<'a, T, I, M> IterMut<'a, T, I> for Trans<M> where
    I: Iterator<T>,
    M: IterMut<'a, T, I>,
{
    fn iter_mut(&'a mut self) -> I {
        self.0.iter_mut()
    }
}

impl<M> Matrix for Trans<M> where M: Matrix {
    fn ncols(&self) -> uint {
        self.0.nrows()
    }

    fn nrows(&self) -> uint {
        self.0.ncols()
    }

    fn size(&self) -> (uint, uint) {
        let (nrows, ncols) = self.0.size();

        (ncols, nrows)
    }
}

impl<'a, V, M> MatrixCol<'a, V> for Trans<M> where M: MatrixRow<'a, V> {
    fn col(&'a self, col: uint) -> ::Result<Col<V>> {
        self.0.row(col).map(|row| Col(row.0))
    }

    unsafe fn unsafe_col(&'a self, col: uint) -> Col<V> {
        Col(self.0.unsafe_row(col).0)
    }
}

impl<'a, V, M> MatrixColMut<'a, V> for Trans<M> where M: MatrixRowMut<'a, V> {
    fn col_mut(&'a mut self, col: uint) -> ::Result<Col<V>> {
        self.0.row_mut(col).map(|row| Col(row.0))
    }

    unsafe fn unsafe_col_mut(&'a mut self, col: uint) -> Col<V> {
        Col(self.0.unsafe_row_mut(col).0)
    }
}

impl<'a, M> MatrixCols<'a> for Trans<M> where M: Matrix {}

impl<T, M> MatrixDiag<T> for Trans<M> where M: MatrixDiag<T> {
    fn diag(&self, diag: int) -> ::Result<Diag<strided::Slice<T>>> {
        self.0.diag(-diag)
    }
}

impl<T, M> MatrixDiagMut<T> for Trans<M> where M: MatrixDiagMut<T> {
    fn diag_mut(&mut self, diag: int) -> ::Result<Diag<strided::MutSlice<T>>> {
        self.0.diag_mut(-diag)
    }
}

impl<'a, M> MatrixMutCols<'a> for Trans<M> where M: Matrix {}

impl<'a, M> MatrixMutRows<'a> for Trans<M> where M: Matrix {}

impl<'a, V, M> MatrixRow<'a, V> for Trans<M> where M: MatrixCol<'a, V> {
    fn row(&'a self, row: uint) -> ::Result<Row<V>> {
        self.0.col(row).map(|col| Row(col.0))
    }

    unsafe fn unsafe_row(&'a self, row: uint) -> Row<V> {
        Row(self.0.unsafe_col(row).0)
    }
}

impl<'a, V, M> MatrixRowMut<'a, V> for Trans<M> where M: MatrixColMut<'a, V> {
    fn row_mut(&'a mut self, row: uint) -> ::Result<Row<V>> {
        self.0.col_mut(row).map(|col| Row(col.0))
    }

    unsafe fn unsafe_row_mut(&'a mut self, row: uint) -> Row<V> {
        Row(self.0.unsafe_col_mut(row).0)
    }
}

impl<'a, M> MatrixRows<'a> for Trans<M> where M: Matrix {}

impl<'a, M, V> Slice<'a, (uint, uint), Trans<V>> for Trans<M> where
    M: Matrix + Slice<'a, (uint, uint), V>,

{
    fn slice(&'a self, start: (uint, uint), end: (uint, uint)) -> ::Result<Trans<V>> {
        let (end_row, end_col) = end;
        let (start_row, start_col) = start;

        self.0.slice((start_col, start_row), (end_col, end_row)).map(|m| Trans(m))
    }

    fn slice_from(&'a self, start: (uint, uint)) -> ::Result<Trans<V>> {
        let end = self.size();

        Slice::slice(self, start, end)
    }

    fn slice_to(&'a self, end: (uint, uint)) -> ::Result<Trans<V>> {
        Slice::slice(self, (0, 0), end)
    }
}

impl<'a, M, V> SliceMut<'a, (uint, uint), Trans<V>> for Trans<M> where
    M: Matrix + SliceMut<'a, (uint, uint), V>,

{
    fn slice_mut(&'a mut self, start: (uint, uint), end: (uint, uint)) -> ::Result<Trans<V>> {
        let (end_row, end_col) = end;
        let (start_row, start_col) = start;

        self.0.slice_mut((start_col, start_row), (end_col, end_row)).map(|m| Trans(m))
    }

    fn slice_from_mut(&'a mut self, start: (uint, uint)) -> ::Result<Trans<V>> {
        let end = self.size();

        SliceMut::slice_mut(self, start, end)
    }

    fn slice_to_mut(&'a mut self, end: (uint, uint)) -> ::Result<Trans<V>> {
        SliceMut::slice_mut(self, (0, 0), end)
    }
}

impl<M> Transpose<M> for Trans<M> {
    fn t(self) -> M {
        self.0
    }
}
