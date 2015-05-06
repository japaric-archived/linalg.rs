//! Extension traits
//!
//!  WARNING! At this point in time, these traits are *not* meant to be used for generic
//!  programming, and will remain unstable. What's guaranteed to be (somewhat) stable is the
//!  functionality, i.e. the methods, provided by them.

use {
    Col, ColMut, Cols, ColsMut, Diag, DiagMut, HStripes, HStripesMut, Row, RowMut, Rows, RowsMut,
    VStripes, VStripesMut, SubMat, SubMatMut,
};

/// Force evaluation of lazy operations
pub trait Eval {
    /// The output of the operation
    type Output;

    /// Evaluates the lazy operation
    fn eval(self) -> Self::Output;
}

/// "Immutable" horizontal splitting
pub trait HSplit: Matrix {
    /// Splits a matrix horizontally at the `i`th row in two immutable pieces
    fn hsplit_at(&self, i: u32) -> (SubMat<Self::Elem>, SubMat<Self::Elem>);
}

/// "Mutable" horizontal splitting
pub trait HSplitMut: HSplit {
    /// Splits a matrix horizontally at the `i`th row in two mutable pieces
    fn hsplit_at_mut(&mut self, u32) -> (SubMatMut<Self::Elem>, SubMatMut<Self::Elem>);
}

/// "Immutable iteration" over a matrix
pub trait Iter<'a>: Matrix {
    /// The iterator
    type Iter: Iterator;

    /// Returns an iterator that yields immutable references to the elements of the matrix
    ///
    /// NOTE For optimization reasons the iteration order is left unspecified, so don't rely on it
    fn iter(&'a self) -> Self::Iter;
}

/// "Mutable iteration" over a matrix
pub trait IterMut<'a>: Iter<'a> {
    /// The iterator
    type IterMut: Iterator;

    /// Returns an iterator that yields mutable references to the elements of the matrix
    ///
    /// NOTE For optimization reasons the iteration order is left unspecified, so don't rely on it
    fn iter_mut(&'a mut self) -> Self::IterMut;
}

/// The basic idea of a matrix: A rectangular array arranged in rows and columns
pub trait Matrix: Sized {
    /// The type of the elements contained in the matrix
    type Elem;

    /// Returns the number of columns the matrix has
    fn ncols(&self) -> u32 {
        self.size().1
    }

    /// Returns the number of rows the matrix has
    fn nrows(&self) -> u32 {
        self.size().0
    }

    /// Returns the size of the matrix
    fn size(&self) -> (u32, u32) {
        (self.nrows(), self.ncols())
    }
}

/// Immutable view into the column of a matrix
pub trait MatrixCol: Matrix {
    /// Returns an immutable view into the `i`th column of the matrix
    fn col(&self, u32) -> Col<Self::Elem>;
}

/// Mutable access to the column of a matrix
pub trait MatrixColMut: MatrixCol {
    /// Returns a mutable "view" into the `i`th column of the matrix
    fn col_mut(&mut self, i: u32) -> ColMut<Self::Elem> {
        ColMut(self.col(i))
    }
}

/// Immutable column-by-column iteration
pub trait MatrixCols: Matrix {
    /// Returns an iterator that yields immutable view into the columns of the matrix
    fn cols(&self) -> Cols<Self::Elem>;
}

/// Mutable column-by-column iteration
pub trait MatrixColsMut: MatrixCols {
    /// Returns an iterator that yields mutable "views" into the columns of the matrix
    fn cols_mut(&mut self) -> ColsMut<Self::Elem> {
        ColsMut(self.cols())
    }
}

/// Immutable view into the diagonal of a matrix
pub trait MatrixDiag: Matrix {
    /// Returns an immutable view into the `i`th diagonal of the matrix
    fn diag(&self, i32) -> Diag<Self::Elem>;
}

/// Mutable access to the diagonal of a matrix
pub trait MatrixDiagMut: MatrixDiag {
    /// Returns a mutable "view" into the `i`th diagonal of the matrix
    fn diag_mut(&mut self, i: i32) -> DiagMut<Self::Elem> {
        DiagMut(self.diag(i))
    }
}

/// "Immutable iteration" over a matrix in horizontal stripes
pub trait MatrixHStripes: Matrix {
    /// Returns an immutable iterator that yields horizontal stripes of `size` rows
    fn hstripes(&self, size: u32) -> HStripes<Self::Elem>;
}

/// "Mutable iteration" over a matrix in horizontal stripes
pub trait MatrixHStripesMut: MatrixHStripes {
    /// Returns a mutable iterator that yields horizontal stripes of `size` rows
    fn hstripes_mut(&mut self, size: u32) -> HStripesMut<Self::Elem> {
        HStripesMut(self.hstripes(size))
    }
}

/// Matrix inverse
pub trait MatrixInverse {
    /// The inversed matrix
    type Output;

    /// Returns the inverse of the input matrix
    fn inv(self) -> Self::Output;
}

/// Immutable view into the row of a matrix
pub trait MatrixRow: Matrix {
    /// Returns an immutable "view" into the `i`th row of the matrix
    fn row(&self, u32) -> Row<Self::Elem>;
}

/// Mutable access to the row of a matrix
pub trait MatrixRowMut: MatrixRow {
    /// Returns a mutable "view" into the `i`th row of the matrix
    fn row_mut(&mut self, i: u32) -> RowMut<Self::Elem> {
        RowMut(self.row(i))
    }
}

/// Immutable row-by-row iteration
pub trait MatrixRows: Matrix {
    /// Returns an iterator that yields immutable views into the rows of a matrix
    fn rows(&self) -> Rows<Self::Elem>;
}

/// Mutable row-by-row iteration
pub trait MatrixRowsMut: MatrixRows {
    /// Returns an iterator that yields mutable "views" into the rows a matrix
    fn rows_mut(&mut self) -> RowsMut<Self::Elem> {
        RowsMut(self.rows())
    }
}

/// Alternative to `IndexSet` (which doesn't exist)
///
/// Usage: `a.col_mut(1).set(b.col(0))`
///
/// Hopefully in the future this will be replaced with `a[(.., 1)] = &b[(.., 0)]`
pub trait Set<T> {
    /// Copies `RHS` into `self`
    fn set(&mut self, rhs: T);
}

/// "Immutable slicing"
pub trait Slice<'a, Range> {
    /// A immutable "slice" of the collection
    type Output;

    /// Returns an immutable view into a "slice" of the collection that spans `Range`
    fn slice(&'a self, Range) -> Self::Output;
}

/// "Mutable slicing"
pub trait SliceMut<'a, Range> {
    /// A mutable "slice" of the collection
    type Output;

    /// Returns a mutable "view" into a "slice" of the collection that spans `Range`
    fn slice_mut(&'a mut self, Range) -> Self::Output;
}

/// The transpose operator
pub trait Transpose {
    /// The transposed data
    type Output;

    /// Returns the transpose of the input
    fn t(self) -> Self::Output;
}

/// "Immutable" vertical splitting
pub trait VSplit: Matrix {
    /// Splits a matrix vertically at the `i`th column in two immutable pieces
    fn vsplit_at(&self, u32) -> (SubMat<Self::Elem>, SubMat<Self::Elem>);
}

/// "Mutable" vertical splitting
pub trait VSplitMut: VSplit {
    /// Splits a matrix vertically at the `i`th column in two mutable pieces
    fn vsplit_at_mut(&mut self, u32) -> (SubMatMut<Self::Elem>, SubMatMut<Self::Elem>);
}

/// "Immutable iteration" over a matrix in vertical stripes
pub trait MatrixVStripes: Matrix {
    /// Returns an immutable iterator that yields vertical stripes of `size` columns
    fn vstripes(&self, size: u32) -> VStripes<Self::Elem>;
}

/// "Mutable iteration" over a matrix in vertical stripes
pub trait MatrixVStripesMut: MatrixVStripes {
    /// Returns a "mutable" iterator that yields horizontal stripes of `size` columns
    fn vstripes_mut(&mut self, size: u32) -> VStripesMut<Self::Elem> {
        VStripesMut(self.vstripes(size))
    }
}
