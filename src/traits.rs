//! Traits

use notsafe::{UnsafeMatrixCol, UnsafeMatrixMutCol, UnsafeMatrixMutRow, UnsafeMatrixRow};
use strided;
use {Col, Cols, Diag, MutCols, MutRows, Row, Rows};

// FIXME (rust-lang/rust#5992) Use trait provided by the standard library
/// The `+=` operator
pub trait AddAssign<R> {
    fn add_assign(&mut self, rhs: &R);
}

/// Immutable iteration over a collection
pub trait Iter<'a, T, I: Iterator<T>> {
    fn iter(&'a self) -> I;
}

/// Immutable view on a column
pub trait MatrixCol<'a, D>: Matrix + UnsafeMatrixCol<'a, D> {
    fn col(&'a self, col: uint) -> Option<Col<D>> {
        if col < self.ncols() {
            Some(unsafe { self.unsafe_col(col) })
        } else {
            None
        }
    }
}

/// Immutable column-by-column iteration
pub trait MatrixCols<'a>: Matrix {
    fn cols(&'a self) -> Cols<Self> {
        Cols {
            mat: self,
            state: 0,
            stop: self.ncols(),
        }
    }
}

/// Immutable view on a diagonal
pub trait MatrixDiag<T> {
    fn diag<'a>(&'a self, diag: int) -> Option<Diag<strided::Slice<'a, T>>>;
}

/// Mutable column access
pub trait MatrixMutCol<'a, D>: Matrix + UnsafeMatrixMutCol<'a, D> {
    fn mut_col(&'a mut self, col: uint) -> Option<Col<D>> {
        if col < self.ncols() {
            Some(unsafe { self.unsafe_mut_col(col) })
        } else {
            None
        }
    }
}

/// Mutable column-by-column iteration
pub trait MatrixMutCols<'a>: Matrix {
    fn mut_cols(&'a mut self) -> MutCols<'a, Self> {
        MutCols {
            stop: self.ncols(),
            mat: self,
            state: 0,
        }
    }
}

/// Mutable diagonal access
pub trait MatrixMutDiag<T> {
    fn mut_diag<'a>(&'a mut self, diag: int) -> Option<Diag<strided::MutSlice<'a, T>>>;
}

/// Mutable row access
pub trait MatrixMutRow<'a, D>: Matrix + UnsafeMatrixMutRow<'a, D> {
    fn mut_row(&'a mut self, row: uint) -> Option<Row<D>> {
        if row < self.nrows() {
            Some(unsafe { self.unsafe_mut_row(row) })
        } else {
            None
        }
    }
}

/// Mutable row-by-row iteration
pub trait MatrixMutRows<'a>: Matrix {
    fn mut_rows(&'a mut self) -> MutRows<'a, Self> {
        MutRows {
            stop: self.nrows(),
            mat: self,
            state: 0,
        }
    }
}

/// Immutable row access
pub trait MatrixRow<'a, D>: Matrix + UnsafeMatrixRow<'a, D> {
    fn row(&'a self, row: uint) -> Option<Row<D>> {
        if row < self.nrows() {
            Some(unsafe { self.unsafe_row(row) })
        } else {
            None
        }
    }
}

/// Immutable row-by-row iteration
pub trait MatrixRows<'a>: Matrix {
    fn rows(&'a self) -> Rows<'a, Self> {
        Rows {
            mat: self,
            state: 0,
            stop: self.nrows(),
        }
    }
}

/// The basic idea of a matrix: A rectangular array of numbers arranged in rows and columns
pub trait Matrix {
    fn ncols(&self) -> uint {
        self.size().1
    }

    fn nrows(&self) -> uint {
        self.size().0
    }

    fn size(&self) -> (uint, uint) {
        (self.nrows(), self.ncols())
    }
}

// FIXME (rust-lang/rust#5992) Use trait provided by the standard library
/// The `*=` operator
pub trait MulAssign<R> {
    fn mul_assign(&mut self, rhs: &R);
}

/// Mutable iteration over a collection
pub trait MutIter<'a, T, I: Iterator<T>> {
    fn mut_iter(&'a mut self) -> I;
}

/// Immutable indexing
pub trait OptionIndex<I, R> {
    fn at<'a>(&'a self, index: &I) -> Option<&'a R>;
}

/// Mutable indexing
pub trait OptionIndexMut<I, R> {
    fn at_mut<'a>(&'a mut self, index: &I) -> Option<&'a mut R>;
}

/// Mutable slicing
pub trait OptionMutSlice<'a, I, S> {
    fn mut_slice(&'a mut self, start: I, end: I) -> Option<S>;
}

/// Immutable slicing
pub trait OptionSlice<'a, I, S> {
    fn slice(&'a self, start: I, end: I) -> Option<S>;
}

// FIXME (rust-lang/rust#5992) Use trait provided by the standard library
/// The `-=` operator
pub trait SubAssign<R> {
    fn sub_assign(&mut self, rhs: &R);
}

/// Consume a column-by-column iterator and return the sum of the columns
pub trait SumCols<T> {
    /// - Memory: `O(nrows)`
    /// - Time: `O(nrows * ncols)`
    fn sum(self) -> Option<Col<Vec<T>>>;
}

/// Consume a row-by-row iterator and return the sum of the rows
pub trait SumRows<T> {
    /// - Memory: `O(ncols)`
    /// - Time: `O(nrows * ncols)`
    fn sum(self) -> Option<Row<Vec<T>>>;
}

/// Creates an owned version from a view
pub trait ToOwned<T> {
    fn to_owned(&self) -> T;
}

/// The transpose operator
pub trait Transpose<T> {
    fn t(self) -> T;
}
