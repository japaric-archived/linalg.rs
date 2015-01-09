use {Col, Diag, Mat, MutCol, MutDiag, MutRow, MutView, Result, Row, Trans, View};
use error::OutOfBounds;
use traits::{
    At, AtMut, Iter, IterMut, Matrix, MatrixCol, MatrixColMut, MatrixCols, MatrixDiag,
    MatrixDiagMut, MatrixMutCols, MatrixMutRows, MatrixRow, MatrixRowMut, MatrixRows, Slice,
    SliceMut, Transpose,
};

impl<T, M> At<(usize, usize)> for Trans<M> where M: At<(usize, usize), Output=T> {
    type Output = T;

    fn at(&self, (row, col): (usize, usize)) -> ::std::result::Result<&T, OutOfBounds> {
        self.0.at((col, row))
    }
}

impl<T, M> AtMut<(usize, usize)> for Trans<M> where M: AtMut<(usize, usize), Output=T> {
    type Output = T;

    fn at_mut(
        &mut self,
        (row, col): (usize, usize),
    ) -> ::std::result::Result<&mut T, OutOfBounds> {
        self.0.at_mut((col, row))
    }
}

impl<'a, I, M> Iter<'a> for Trans<M> where
    I: Iterator,
    M: Iter<'a, Iter=I>,
{
    type Iter = I;

    fn iter(&'a self) -> I {
        self.0.iter()
    }
}

impl<'a, I, M> IterMut<'a> for Trans<M> where
    I: Iterator,
    M: IterMut<'a, Iter=I>,
{
    type Iter = I;

    fn iter_mut(&'a mut self) -> I {
        self.0.iter_mut()
    }
}

impl<T, M> Matrix for Trans<M> where M: Matrix<Elem=T> {
    type Elem = T;

    fn ncols(&self) -> usize {
        self.0.nrows()
    }

    fn nrows(&self) -> usize {
        self.0.ncols()
    }

    fn size(&self) -> (usize, usize) {
        let (nrows, ncols) = self.0.size();

        (ncols, nrows)
    }
}

impl<T, M> MatrixCol for Trans<M> where M: MatrixRow<Elem=T> {
    fn col(&self, col: usize) -> Result<Col<T>> {
        self.0.row(col).map(|row| Col(row.0))
    }

    unsafe fn unsafe_col(&self, col: usize) -> Col<T> {
        Col(self.0.unsafe_row(col).0)
    }
}

impl<T, M> MatrixColMut for Trans<M> where M: MatrixRowMut<Elem=T> {
    fn col_mut(&mut self, col: usize) -> Result<MutCol<T>> {
        self.0.row_mut(col).map(|row| MutCol(row.0))
    }

    unsafe fn unsafe_col_mut(&mut self, col: usize) -> MutCol<T> {
        MutCol(self.0.unsafe_row_mut(col).0)
    }
}

impl<M> MatrixCols for Trans<M> where M: Matrix {}

impl<T, M> MatrixDiag for Trans<M> where M: MatrixDiag<Elem=T> {
    fn diag(&self, diag: isize) -> Result<Diag<T>> {
        self.0.diag(-diag)
    }
}

impl<T, M> MatrixDiagMut for Trans<M> where M: MatrixDiagMut<Elem=T> {
    fn diag_mut(&mut self, diag: isize) -> Result<MutDiag<T>> {
        self.0.diag_mut(-diag)
    }
}

impl<M> MatrixMutCols for Trans<M> where M: Matrix {}

impl<M> MatrixMutRows for Trans<M> where M: Matrix {}

impl<T, M> MatrixRow for Trans<M> where M: MatrixCol<Elem=T> {
    fn row(&self, row: usize) -> Result<Row<T>> {
        self.0.col(row).map(|col| Row(col.0))
    }

    unsafe fn unsafe_row(&self, row: usize) -> Row<T> {
        Row(self.0.unsafe_col(row).0)
    }
}

impl<T, M> MatrixRowMut for Trans<M> where M: MatrixColMut<Elem=T> {
    fn row_mut(&mut self, row: usize) -> Result<MutRow<T>> {
        self.0.col_mut(row).map(|col| MutRow(col.0))
    }

    unsafe fn unsafe_row_mut(&mut self, row: usize) -> MutRow<T> {
        MutRow(self.0.unsafe_col_mut(row).0)
    }
}

impl<M> MatrixRows for Trans<M> where M: Matrix {}

impl<'a, M, V> Slice<'a, (usize, usize)> for Trans<M> where
    M: Matrix + Slice<'a, (usize, usize), Slice=V>,
{
    type Slice = Trans<V>;

    fn slice(&'a self, start: (usize, usize), end: (usize, usize)) -> ::Result<Trans<V>> {
        let (end_row, end_col) = end;
        let (start_row, start_col) = start;

        self.0.slice((start_col, start_row), (end_col, end_row)).map(Trans)
    }

    fn slice_from(&'a self, start: (usize, usize)) -> ::Result<Trans<V>> {
        let end = self.size();

        Slice::slice(self, start, end)
    }

    fn slice_to(&'a self, end: (usize, usize)) -> ::Result<Trans<V>> {
        Slice::slice(self, (0, 0), end)
    }
}

impl<'a, M, V> SliceMut<'a, (usize, usize)> for Trans<M> where
    M: Matrix + SliceMut<'a, (usize, usize), Slice=V>,
{
    type Slice = Trans<V>;

    fn slice_mut(&'a mut self, start: (usize, usize), end: (usize, usize)) -> ::Result<Trans<V>> {
        let (end_row, end_col) = end;
        let (start_row, start_col) = start;

        self.0.slice_mut((start_col, start_row), (end_col, end_row)).map(Trans)
    }

    fn slice_from_mut(&'a mut self, start: (usize, usize)) -> ::Result<Trans<V>> {
        let end = self.size();

        SliceMut::slice_mut(self, start, end)
    }

    fn slice_to_mut(&'a mut self, end: (usize, usize)) -> ::Result<Trans<V>> {
        SliceMut::slice_mut(self, (0, 0), end)
    }
}

impl<T> Transpose for Mat<T> {
    type Output = Trans<Mat<T>>;

    fn t(self) -> Trans<Mat<T>> {
        Trans(self)
    }
}

impl<'a, T> Transpose for MutView<'a, T> {
    type Output = Trans<MutView<'a, T>>;

    fn t(self) -> Trans<MutView<'a, T>> {
        Trans(self)
    }
}

impl<'a, T> Transpose for View<'a, T> {
    type Output = Trans<View<'a, T>>;

    fn t(self) -> Trans<View<'a, T>> {
        Trans(self)
    }
}

impl<M> Transpose for Trans<M> {
    type Output = M;

    fn t(self) -> M {
        self.0
    }
}
