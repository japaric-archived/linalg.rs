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

impl<M: Collection> Collection for Trans<M> {
    fn len(&self) -> uint {
        self.mat.len()
    }
}

impl<T, M: Index<(uint, uint), T>> Index<(uint, uint), T> for Trans<M> {
    fn index(&self, &(row, col): &(uint, uint)) -> &T {
        self.mat.index(&(col, row))
    }
}

impl<T, M: IndexMut<(uint, uint), T>> IndexMut<(uint, uint), T> for Trans<M> {
    fn index_mut(&mut self, &(row, col): &(uint, uint)) -> &mut T {
        self.mat.index_mut(&(col, row))
    }
}

impl<'a, T, I: Iterator<T>, M: Iter<'a, T, I>> Iter<'a, T, I> for Trans<M> {
    fn iter(&'a self) -> I {
        self.mat.iter()
    }
}

impl<M: Matrix> Matrix for Trans<M> {
    fn ncols(&self) -> uint {
        self.mat.nrows()
    }

    fn nrows(&self) -> uint {
        self.mat.ncols()
    }

    fn size(&self) -> (uint, uint) {
        let (nrows, ncols) = self.mat.size();

        (ncols, nrows)
    }
}

impl<'a, D, M: MatrixRow<'a, D>> MatrixCol<'a, D> for Trans<M> {
    fn col(&'a self, col: uint) -> Option<Col<D>> {
        self.mat.row(col).map(|row| Col { data: row.data })
    }
}

impl<'a, M: Matrix> MatrixCols<'a> for Trans<M> {}

impl<T, M: MatrixDiag<T>> MatrixDiag<T> for Trans<M> {
    fn diag(&self, diag: int) -> Option<Diag<strided::Slice<T>>> {
        self.mat.diag(-diag)
    }
}

impl<'a, D, M: MatrixMutRow<'a, D>> MatrixMutCol<'a, D> for Trans<M> {
    fn mut_col(&'a mut self, col: uint) -> Option<Col<D>> {
        self.mat.mut_row(col).map(|row| Col { data: row.data })
    }
}

impl<'a, M: Matrix> MatrixMutCols<'a> for Trans<M> {}

impl<T, M: MatrixMutDiag<T>> MatrixMutDiag<T> for Trans<M> {
    fn mut_diag(&mut self, diag: int) -> Option<Diag<strided::MutSlice<T>>> {
        self.mat.mut_diag(-diag)
    }
}

impl<'a, D, M: MatrixMutCol<'a, D>> MatrixMutRow<'a, D> for Trans<M> {
    fn mut_row(&'a mut self, row: uint) -> Option<Row<D>> {
        self.mat.mut_col(row).map(|col| Row { data: col.data })
    }
}

impl<'a, M: Matrix> MatrixMutRows<'a> for Trans<M> {}

impl<'a, D, M: MatrixCol<'a, D>> MatrixRow<'a, D> for Trans<M> {
    fn row(&'a self, row: uint) -> Option<Row<D>> {
        self.mat.col(row).map(|col| Row { data: col.data })
    }
}

impl<'a, M: Matrix> MatrixRows<'a> for Trans<M> {}

impl<'a, T, I: Iterator<T>, M: MutIter<'a, T, I>> MutIter<'a, T, I> for Trans<M> {
    fn mut_iter(&'a mut self) -> I {
        self.mat.mut_iter()
    }
}

impl<T, M: OptionIndex<(uint, uint), T>> OptionIndex<(uint, uint), T> for Trans<M> {
    fn at(&self, &(row, col): &(uint, uint)) -> Option<&T> {
        self.mat.at(&(col, row))
    }
}

impl<T, M: OptionIndexMut<(uint, uint), T>> OptionIndexMut<(uint, uint), T> for Trans<M> {
    fn at_mut(&mut self, &(row, col): &(uint, uint)) -> Option<&mut T> {
        self.mat.at_mut(&(col, row))
    }
}

impl<'a, V, M: OptionMutSlice<'a, (uint, uint), V>> OptionMutSlice<'a, (uint, uint), Trans<V>>
for Trans<M> {
    fn mut_slice(&'a mut self, start: (uint, uint), end: (uint, uint)) -> Option<Trans<V>> {
        let (end_row, end_col) = end;
        let (start_row, start_col) = start;

        self.mat.mut_slice((start_col, start_row), (end_col, end_row)).map(|m| Trans { mat: m })
    }
}

impl<'a, V, M: OptionSlice<'a, (uint, uint), V>> OptionSlice<'a, (uint, uint), Trans<V>>
for Trans<M> {
    fn slice(&'a self, start: (uint, uint), end: (uint, uint)) -> Option<Trans<V>> {
        let (end_row, end_col) = end;
        let (start_row, start_col) = start;

        self.mat.slice((start_col, start_row), (end_col, end_row)).map(|m| Trans { mat: m })
    }
}

impl<D> Transpose<Row<D>> for Col<D> {
    fn t(self) -> Row<D> {
        Row {
            data: self.data,
        }
    }
}

impl<D> Transpose<Col<D>> for Row<D> {
    fn t(self) -> Col<D> {
        Col {
            data: self.data,
        }
    }
}

impl<'a, M> Transpose<M> for Trans<M> {
    fn t(self) -> M {
        self.mat
    }
}

impl<'a, D, M: UnsafeMatrixRow<'a, D>> UnsafeMatrixCol<'a, D> for Trans<M> {
    unsafe fn unsafe_col(&'a self, col: uint) -> Col<D> {
        Col {
            data: self.mat.unsafe_row(col).data,
        }
    }
}

impl<'a, D, M: UnsafeMatrixMutRow<'a, D>> UnsafeMatrixMutCol<'a, D> for Trans<M> {
    unsafe fn unsafe_mut_col(&'a mut self, col: uint) -> Col<D> {
        Col {
            data: self.mat.unsafe_mut_row(col).data,
        }
    }
}

impl<'a, D, M: UnsafeMatrixMutCol<'a, D>> UnsafeMatrixMutRow<'a, D> for Trans<M> {
    unsafe fn unsafe_mut_row(&'a mut self, row: uint) -> Row<D> {
        Row {
            data: self.mat.unsafe_mut_col(row).data,
        }
    }
}

impl<'a, D, M: UnsafeMatrixCol<'a, D>> UnsafeMatrixRow<'a, D> for Trans<M> {
    unsafe fn unsafe_row(&'a self, row: uint) -> Row<D> {
        Row {
            data: self.mat.unsafe_col(row).data,
        }
    }
}
