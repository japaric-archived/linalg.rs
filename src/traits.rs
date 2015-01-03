//! Traits

use error::OutOfBounds;
use {Col, Cols, Diag, Error, MutCol, MutCols, MutDiag, MutRow, MutRows, Result, Row, Rows};

/// The `+=` operator
// FIXME (rust-lang/rfcs#393) Use trait provided by the standard library
pub trait AddAssign<R> {
    /// Performs the operation `self += rhs`
    ///
    /// **Note** The operator sugar has yet to be implemented. See rust-lang/rfcs#393
    fn add_assign(&mut self, rhs: R);
}

/// Bounds-checked immutable indexing
// FIXME (AI) `T` should be an associated type
pub trait At<I, T> for Sized? {
    /// Returns an immutable reference to the element at the given index
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the index is out of bounds
    fn at(&self, index: I) -> ::std::result::Result<&T, OutOfBounds>;
}

/// Bounds-checked mutable indexing
// FIXME (AI) `T` should be an associated type
pub trait AtMut<I, T> for Sized? {
    /// Returns a mutable reference to the element at the given index
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the index is out of bounds
    fn at_mut(&mut self, index: I) -> ::std::result::Result<&mut T, OutOfBounds>;
}

/// Immutable iteration over a collection
// FIXME (AI) `'a`, `T`, `I` should be associated items
pub trait Iter<'a, T, I: Iterator<T>> {
    /// Returns an iterator that yields immutable references to the elements of the collection
    fn iter(&'a self) -> I;
}

/// Mutable iteration over a collection
// FIXME (AI) `'a`, `T`, `I` should be associated items
pub trait IterMut<'a, T, I: Iterator<T>> {
    /// Returns an iterator that yields mutable references to the elements of the collection
    fn iter_mut(&'a mut self) -> I;
}

/// The basic idea of a matrix: A rectangular array arranged in rows and columns
pub trait Matrix: Sized {
    /// Returns the number of columns the matrix has
    fn ncols(&self) -> uint {
        self.size().1
    }

    /// Returns the number of rows the matrix has
    fn nrows(&self) -> uint {
        self.size().0
    }

    /// Returns the size of the matrix
    fn size(&self) -> (uint, uint) {
        (self.nrows(), self.ncols())
    }
}

/// Immutable view on a column
// FIXME (AI) `T` should be an associated type
pub trait MatrixCol<T>: Matrix {
    /// Returns an immutable view into the column at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchColumn` if the index is out of bounds
    fn col(&self, col: uint) -> Result<Col<T>> {
        if col < self.ncols() {
            Ok(unsafe { self.unsafe_col(col) })
        } else {
            Err(Error::NoSuchColumn)
        }
    }

    /// Returns a view into the column at the given index without performing bounds checking
    unsafe fn unsafe_col(&self, col: uint) -> Col<T>;
}

/// Mutable access to a column
// FIXME (AI) `T` should be an associated type
pub trait MatrixColMut<T>: MatrixCol<T> {
    /// Returns a mutable view into the column at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchColumn` if the index is out of bounds
    fn col_mut(&mut self, col: uint) -> Result<MutCol<T>> {
        if col < self.ncols() {
            Ok(unsafe { self.unsafe_col_mut(col) })
        } else {
            Err(Error::NoSuchColumn)
        }
    }

    /// Returns a mutable view into the column at the given index without performing bounds
    /// checking
    unsafe fn unsafe_col_mut(&mut self, col: uint) -> MutCol<T>;
}

/// Immutable column-by-column iteration
pub trait MatrixCols: Matrix {
    /// Returns an iterator that yields immutable views into the columns of the matrix
    fn cols(&self) -> Cols<Self> {
        Cols(unsafe { ::From::parts(self) })
    }
}

/// Immutable view on a diagonal
// FIXME (AI) `T` should be an associated type
pub trait MatrixDiag<T> {
    /// Returns a view into the diagonal at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchDiagonal` if the index is out of bounds
    fn diag(&self, diag: int) -> Result<Diag<T>>;
}

/// Mutable access to a diagonal
// FIXME (AI) `T` should be an associated type
pub trait MatrixDiagMut<T> {
    /// Returns a mutable view into the diagonal at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchDiagonal` if the index is out of bounds
    fn diag_mut(&mut self, diag: int) -> ::Result<MutDiag<T>>;
}

/// Mutable column-by-column iteration
pub trait MatrixMutCols: Matrix {
    /// Returns an iterator that yields mutable views into the columns of the matrix
    fn mut_cols(&mut self) -> MutCols<Self> {
        MutCols(unsafe { ::From::parts(&*self) })
    }
}

/// Mutable row-by-row iteration
pub trait MatrixMutRows: Matrix {
    /// Returns an iterator that yields mutable views into the rows of the matrix
    fn mut_rows(&mut self) -> MutRows<Self> {
        MutRows(unsafe { ::From::parts(&*self) })
    }
}

/// Immutable view into a row
// FIXME (AI) `T` should be an associated type
pub trait MatrixRow<T>: Matrix {
    /// Returns an immutable view into the row at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchRow` if the index is out of bounds
    fn row(&self, row: uint) -> Result<Row<T>> {
        if row < self.nrows() {
            Ok(unsafe { self.unsafe_row(row) })
        } else {
            Err(Error::NoSuchRow)
        }
    }

    /// Returns an immutable view into the row at the given index without performing bounds
    /// checking
    unsafe fn unsafe_row(&self, row: uint) -> Row<T>;
}

/// Immutable row-by-row iteration
pub trait MatrixRows: Matrix {
    /// Returns an iterator that yields immutable views into each row of the matrix
    fn rows(&self) -> Rows<Self> {
        Rows(unsafe { ::From::parts(self) })
    }
}

/// Mutable access to a row
// FIXME (AI) `T` should be an associated type
pub trait MatrixRowMut<T>: MatrixRow<T> {
    /// Returns a mutable view into the row at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchRow` if the index is out of bounds
    fn row_mut(&mut self, row: uint) -> Result<MutRow<T>> {
        if row < self.nrows() {
            Ok(unsafe { self.unsafe_row_mut(row) })
        } else {
            Err(Error::NoSuchRow)
        }
    }

    /// Returns a mutable view into the row at the given index without performing bounds checking
    unsafe fn unsafe_row_mut(&mut self, row: uint) -> MutRow<T>;
}

/// The `*=` operator
// FIXME (rust-lang/rfcs#393) Use trait provided by the standard library
pub trait MulAssign<R> {
    /// Performs the operation `self *= rhs`
    ///
    /// **Note** The operator sugar has yet to be implemented. See rust-lang/rfcs#393
    fn mul_assign(&mut self, rhs: R);
}

/// A more flexible slicing trait
///
/// *Note* Sadly this doesn't have operator sugar. You won't be able to use the slicing operator
/// `[]` with this library until Rust gets HKT.
// FIXME (AI) `'a`, `R` should be associated items
pub trait Slice<'a, I, S> {
    /// Returns an immutable view into a fraction of the collection that spans `start` : `end`
    fn slice(&'a self, start: I, end: I) -> ::Result<S>;
    /// Convenience method for `slice(start, end_of_collection)`
    fn slice_from(&'a self, start: I) -> ::Result<S>;
    /// Convenience method for `slice(start_of_collection, end)`
    fn slice_to(&'a self, end: I) -> ::Result<S>;
}

/// Mutable version of the `Slice` trait
// FIXME (AI) `'a`, `R` should be associated items
pub trait SliceMut<'a, I, S> {
    /// Returns a mutable view into a fraction of the collection that spans `start` : `end`
    fn slice_mut(&'a mut self, start: I, end: I) -> ::Result<S>;
    /// Convenience method for `slice_mut(start, end_of_collection)`
    fn slice_from_mut(&'a mut self, start: I) -> ::Result<S>;
    /// Convenience method for `slice_mut(start_of_collection, end)`
    fn slice_to_mut(&'a mut self, end: I) -> ::Result<S>;
}

/// The `-=` operator
// FIXME (rust-lang/rfcs#393) Use trait provided by the standard library
pub trait SubAssign<R> {
    /// Performs the operation `self -= rhs`
    ///
    /// **Note** The operator sugar has yet to be implemented. See rust-lang/rfcs#393
    fn sub_assign(&mut self, rhs: R);
}

/// Make an owned clone from a view
// TODO (rust-lang/rust#18910) Use trait provided by the standard library
pub trait ToOwned<T> {
    /// Returns an owned clone from the view
    fn to_owned(&self) -> T;
}

/// The transpose operator
// FIXME (AI) `T` should be an associated type
pub trait Transpose<T> {
    /// Returns the transpose of the input
    fn t(self) -> T;
}
