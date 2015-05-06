//! Transposed matrices

mod cols;
mod rows;
mod stripes;

use std::ops::{Index, IndexMut, RangeFull};

use traits::{
    HSplit, HSplitMut, Iter, IterMut, Matrix, MatrixCol, MatrixColMut, MatrixCols, MatrixColsMut,
    MatrixDiag, MatrixDiagMut, MatrixHStripes, MatrixHStripesMut, MatrixRow, MatrixRowMut,
    MatrixRows, MatrixRowsMut, MatrixVStripes, MatrixVStripesMut, Slice, SliceMut, Transpose,
    VSplit, VSplitMut,
};
use {Col, ColMut, Diag, DiagMut, Row, RowMut, Transposed, SubMat, SubMatMut};

/// Iterator over the columns of an immutable transposed matrix
pub struct Cols<'a, T>(::Rows<'a, T>);

/// Iterator over the columns of a mutable transposed matrix
pub struct ColsMut<'a, T>(::RowsMut<'a, T>);

/// An immutable iterator over a transposed matrix in horizontal stripes
pub struct HStripes<'a, T>(::VStripes<'a, T>);

/// A "mutable" iterator over a transposed matrix in horizontal stripes
pub struct HStripesMut<'a, T>(::VStripesMut<'a, T>);

/// Iterator over the rows of an immutable transposed matrix
pub struct Rows<'a, T>(::Cols<'a, T>);

/// Iterator over the rows of a mutable transposed matrix
pub struct RowsMut<'a, T>(::ColsMut<'a, T>);

/// An immutable iterator over a transposed matrix in vertical stripes
pub struct VStripes<'a, T>(::HStripes<'a, T>);

/// A "mutable" iterator over a transposed matrix in vertical stripes
pub struct VStripesMut<'a, T>(::HStripesMut<'a, T>);

impl<M> Transposed<M> {
    /// Returns an immutable "view" into the `i`th column of the matrix
    pub fn col(&self, i: u32) -> Col<M::Elem> where M: MatrixRow {
        Col(self.0.row(i).0)
    }

    /// Returns a mutable "view" into the `i`th column of the matrix
    pub fn col_mut(&mut self, i: u32) -> ColMut<M::Elem> where M: MatrixRowMut {
        ColMut(self.col(i))
    }

    /// Returns an iterator that yields immutable view into the columns of the matrix
    pub fn cols(&self) -> Cols<M::Elem> where M: MatrixRows {
        Cols(self.0.rows())
    }

    /// Returns an iterator that yields mutable "views" into the columns of the matrix
    pub fn cols_mut(&mut self) -> ColsMut<M::Elem> where M: MatrixRowsMut {
        ColsMut(self.0.rows_mut())
    }

    /// Returns an immutable view into the `i`th diagonal of the matrix
    pub fn diag(&self, i: i32) -> Diag<M::Elem> where M: MatrixDiag {
        self.0.diag(-i)
    }

    /// Returns a mutable "view" into the `i`th diagonal of the matrix
    pub fn diag_mut(&mut self, i: i32) -> DiagMut<M::Elem> where M: MatrixDiagMut {
        DiagMut(self.diag(i))
    }

    /// Splits a matrix horizontally at the `i`th row in two immutable pieces
    pub fn hsplit_at(
        &self,
        i: u32,
    ) -> (Transposed<SubMat<M::Elem>>, Transposed<SubMat<M::Elem>>) where
        M: VSplit,
    {
        let (left, right) = self.0.vsplit_at(i);

        (Transposed(left), Transposed(right))
    }

    /// Splits a matrix horizontally at the `i`th row in two mutable pieces
    pub fn hsplit_at_mut(
        &mut self,
        i: u32,
    ) -> (Transposed<SubMatMut<M::Elem>>, Transposed<SubMatMut<M::Elem>>) where
        M: VSplitMut,
    {
        let (left, right) = self.0.vsplit_at_mut(i);

        (Transposed(left), Transposed(right))
    }

    /// Returns an immutable iterator that yields horizontal stripes of `size` rows
    pub fn hstripes(&self, size: u32) -> HStripes<M::Elem> where M: MatrixVStripes {
        HStripes(self.0.vstripes(size))
    }

    /// Returns an "mutable" iterator that yields horizontal stripes of `size` rows
    pub fn hstripes_mut(&mut self, size: u32) -> HStripesMut<M::Elem> where M: MatrixVStripesMut {
        HStripesMut(self.0.vstripes_mut(size))
    }

    /// Returns an iterator that yields immutable references to the elements of the matrix
    ///
    /// NOTE For optimization reasons the iteration order is left unspecified, so don't rely on it
    pub fn iter<'a>(&'a self) -> M::Iter where M: Iter<'a> {
        self.0.iter()
    }

    /// Returns an iterator that yields mutable references to the elements of the matrix
    ///
    /// NOTE For optimization reasons the iteration order is left unspecified, so don't rely on it
    pub fn iter_mut<'a>(&'a mut self) -> M::IterMut where M: IterMut<'a> {
        self.0.iter_mut()
    }

    /// Returns an immutable "view" into the `i`th row of the matrix
    pub fn row(&self, i: u32) -> Row<M::Elem> where M: MatrixCol {
        Row(self.0.col(i).0)
    }

    /// Returns a mutable "view" into the `i`th row of the matrix
    pub fn row_mut(&mut self, i: u32) -> RowMut<M::Elem> where M: MatrixColMut {
        RowMut(self.row(i))
    }

    /// Returns an iterator that yields immutable views into the rows of a matrix
    pub fn rows(&self) -> Rows<M::Elem> where M: MatrixCols {
        Rows(self.0.cols())
    }

    /// Returns an iterator that yields mutable "views" into the rows a matrix
    pub fn rows_mut(&mut self) -> RowsMut<M::Elem> where M: MatrixColsMut {
        RowsMut(self.0.cols_mut())
    }

    /// Splits a matrix vertically at the `i`th column in two immutable pieces
    pub fn vsplit_at(
        &self,
        i: u32,
    ) -> (Transposed<SubMat<M::Elem>>, Transposed<SubMat<M::Elem>>) where
        M: HSplit,
    {
        let (top, bottom) = self.0.hsplit_at(i);

        (Transposed(top), Transposed(bottom))
    }

    /// Splits a matrix vertically at the `i`th column in two mutable pieces
    pub fn vsplit_at_mut(
        &mut self,
        i: u32,
    ) -> (Transposed<SubMatMut<M::Elem>>, Transposed<SubMatMut<M::Elem>>) where
        M: HSplitMut,
    {
        let (top, bottom) = self.0.hsplit_at_mut(i);

        (Transposed(top), Transposed(bottom))
    }

    /// Returns an immutable iterator that yields vertical stripes of `size` columns
    pub fn vstripes(&self, size: u32) -> VStripes<M::Elem> where M: MatrixHStripes {
        VStripes(self.0.hstripes(size))
    }

    /// Returns a "mutable" iterator that yields vertical stripes of `size` columns
    pub fn vstripes_mut(&mut self, size: u32) -> VStripesMut<M::Elem> where M: MatrixHStripesMut {
        VStripesMut(self.0.hstripes_mut(size))
    }

}

impl<T, M> Index<(u32, u32)> for Transposed<M> where M: Index<(u32, u32), Output=T>{
    type Output = T;

    fn index(&self, (row, col): (u32, u32)) -> &T {
        &self.0[(col, row)]
    }
}

impl<T, M> IndexMut<(u32, u32)> for Transposed<M> where M: IndexMut<(u32, u32), Output=T>{
    fn index_mut(&mut self, (row, col): (u32, u32)) -> &mut T {
        &mut self.0[(col, row)]
    }
}

impl<M> Matrix for Transposed<M> where M: Matrix {
    type Elem = M::Elem;

    fn ncols(&self) -> u32 {
        self.0.nrows()
    }

    fn nrows(&self) -> u32 {
        self.0.ncols()
    }
}

impl<'a, M> Slice<'a, RangeFull> for Transposed<M> where
    M: Slice<'a, RangeFull>,
{
    type Output = Transposed<M::Output>;

    fn slice(&'a self, _: RangeFull) -> Transposed<M::Output> {
        Transposed(self.0.slice(..))
    }
}

impl<'a, M> SliceMut<'a, RangeFull> for Transposed<M> where
    M: SliceMut<'a, RangeFull>,
{
    type Output = Transposed<M::Output>;

    fn slice_mut(&'a mut self, _: RangeFull) -> Transposed<M::Output> {
        Transposed(self.0.slice_mut(..))
    }
}

impl<'a, R, C, S, M> Slice<'a, (R, C)> for Transposed<M> where
    M: Slice<'a, (C, R), Output=S>,
    S: Transpose,
{
    type Output = S::Output;

    fn slice(&'a self, (r, c): (R, C)) -> S::Output {
        self.0.slice((c, r)).t()
    }
}

impl<'a, R, C, S, M> SliceMut<'a, (R, C)> for Transposed<M> where
    M: SliceMut<'a, (C, R), Output=S>,
    S: Transpose,
{
    type Output = S::Output;

    fn slice_mut(&'a mut self, (r, c): (R, C)) -> S::Output {
        self.0.slice_mut((c, r)).t()
    }
}

impl<M> Transpose for Transposed<M> {
    type Output = M;

    fn t(self) -> M {
        self.0
    }
}
