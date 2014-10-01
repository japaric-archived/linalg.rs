use strided;
use notsafe::{
    UnsafeMatrixCol,
    UnsafeMatrixMutCol,
    UnsafeMatrixMutRow,
    UnsafeMatrixRow,
};
use traits::{
    Iter,
    Matrix,
    MatrixCol,
    MatrixCols,
    MatrixMutCol,
    MatrixMutCols,
    MatrixMutDiag,
    MatrixMutRow,
    MatrixMutRows,
    MatrixDiag,
    MatrixRow,
    MatrixRows,
    MutIter,
    OptionIndex,
    OptionIndexMut,
    OptionMutSlice,
    OptionSlice,
    Transpose,
};
use {
    Col,
    Diag,
    Row,
    Trans,
};

impl<M> Collection for Trans<M> where M: Collection {
    fn len(&self) -> uint {
        self.0.len()
    }
}

impl<T, M> Index<(uint, uint), T> for Trans<M> where M: Index<(uint, uint), T> {
    fn index(&self, &(row, col): &(uint, uint)) -> &T {
        self.0.index(&(col, row))
    }
}

impl<T, M> IndexMut<(uint, uint), T> for Trans<M> where M: IndexMut<(uint, uint), T> {
    fn index_mut(&mut self, &(row, col): &(uint, uint)) -> &mut T {
        self.0.index_mut(&(col, row))
    }
}

impl<'a, T, I, M> Iter<'a, T, I> for Trans<M> where I: Iterator<T>, M: Iter<'a, T, I> {
    fn iter(&'a self) -> I {
        self.0.iter()
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

impl<'a, D, M> MatrixCol<'a, D> for Trans<M> where M: MatrixRow<'a, D> {
    fn col(&'a self, col: uint) -> Option<Col<D>> {
        self.0.row(col).map(|row| Col(row.0))
    }
}

impl<'a, M> MatrixCols<'a> for Trans<M> where M: Matrix {}

impl<T, M> MatrixDiag<T> for Trans<M> where M: MatrixDiag<T> {
    fn diag(&self, diag: int) -> Option<Diag<strided::Slice<T>>> {
        self.0.diag(-diag)
    }
}

impl<'a, D, M> MatrixMutCol<'a, D> for Trans<M> where M: MatrixMutRow<'a, D> {
    fn mut_col(&'a mut self, col: uint) -> Option<Col<D>> {
        self.0.mut_row(col).map(|row| Col(row.0))
    }
}

impl<'a, M> MatrixMutCols<'a> for Trans<M> where M: Matrix {}

impl<T, M> MatrixMutDiag<T> for Trans<M> where M: MatrixMutDiag<T> {
    fn mut_diag(&mut self, diag: int) -> Option<Diag<strided::MutSlice<T>>> {
        self.0.mut_diag(-diag)
    }
}

impl<'a, D, M> MatrixMutRow<'a, D> for Trans<M> where M: MatrixMutCol<'a, D> {
    fn mut_row(&'a mut self, row: uint) -> Option<Row<D>> {
        self.0.mut_col(row).map(|col| Row(col.0))
    }
}

impl<'a, M> MatrixMutRows<'a> for Trans<M> where M: Matrix {}

impl<'a, D, M> MatrixRow<'a, D> for Trans<M> where M: MatrixCol<'a, D> {
    fn row(&'a self, row: uint) -> Option<Row<D>> {
        self.0.col(row).map(|col| Row(col.0))
    }
}

impl<'a, M> MatrixRows<'a> for Trans<M> where M: Matrix {}

impl<'a, T, I, M> MutIter<'a, T, I> for Trans<M> where I: Iterator<T>, M: MutIter<'a, T, I> {
    fn mut_iter(&'a mut self) -> I {
        self.0.mut_iter()
    }
}

impl<T, M> OptionIndex<(uint, uint), T> for Trans<M> where M: OptionIndex<(uint, uint), T> {
    fn at(&self, &(row, col): &(uint, uint)) -> Option<&T> {
        self.0.at(&(col, row))
    }
}

impl<T, M> OptionIndexMut<(uint, uint), T> for Trans<M> where M: OptionIndexMut<(uint, uint), T> {
    fn at_mut(&mut self, &(row, col): &(uint, uint)) -> Option<&mut T> {
        self.0.at_mut(&(col, row))
    }
}

impl<'a, V, M> OptionMutSlice<'a, (uint, uint), Trans<V>> for Trans<M> where
    M: OptionMutSlice<'a, (uint, uint), V>
{
    fn mut_slice(&'a mut self, start: (uint, uint), end: (uint, uint)) -> Option<Trans<V>> {
        let (end_row, end_col) = end;
        let (start_row, start_col) = start;

        self.0.mut_slice((start_col, start_row), (end_col, end_row)).map(|m| Trans(m))
    }
}

impl<'a, V, M> OptionSlice<'a, (uint, uint), Trans<V>> for Trans<M> where
    M: OptionSlice<'a, (uint, uint), V>
{
    fn slice(&'a self, start: (uint, uint), end: (uint, uint)) -> Option<Trans<V>> {
        let (end_row, end_col) = end;
        let (start_row, start_col) = start;

        self.0.slice((start_col, start_row), (end_col, end_row)).map(|m| Trans(m))
    }
}

impl<D> Transpose<Row<D>> for Col<D> {
    fn t(self) -> Row<D> {
        Row(self.0)
    }
}

impl<D> Transpose<Col<D>> for Row<D> {
    fn t(self) -> Col<D> {
        Col(self.0)
    }
}

impl<'a, M> Transpose<M> for Trans<M> {
    fn t(self) -> M {
        self.0
    }
}

impl<'a, D, M> UnsafeMatrixCol<'a, D> for Trans<M> where M: UnsafeMatrixRow<'a, D> {
    unsafe fn unsafe_col(&'a self, col: uint) -> Col<D> {
        Col(self.0.unsafe_row(col).0)
    }
}

impl<'a, D, M> UnsafeMatrixMutCol<'a, D> for Trans<M> where M: UnsafeMatrixMutRow<'a, D> {
    unsafe fn unsafe_mut_col(&'a mut self, col: uint) -> Col<D> {
        Col(self.0.unsafe_mut_row(col).0)
    }
}

impl<'a, D, M> UnsafeMatrixMutRow<'a, D> for Trans<M> where M: UnsafeMatrixMutCol<'a, D> {
    unsafe fn unsafe_mut_row(&'a mut self, row: uint) -> Row<D> {
        Row(self.0.unsafe_mut_col(row).0)
    }
}

impl<'a, D, M> UnsafeMatrixRow<'a, D> for Trans<M> where M: UnsafeMatrixCol<'a, D> {
    unsafe fn unsafe_row(&'a self, row: uint) -> Row<D> {
        Row(self.0.unsafe_col(row).0)
    }
}
