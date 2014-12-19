//! Traits

use error::OutOfBounds;
use {Col, Cols, Diag, Error, MutCols, MutRows, Row, Rows, strided};

/// The `+=` operator
// FIXME (rust-lang/rfcs#393) Use trait provided by the standard library
pub trait AddAssign<R> {
    /// Performs the operation `self += rhs`
    ///
    /// **Note** The operator sugar has yet to be implemented. See rust-lang/rfcs#393
    fn add_assign(&mut self, rhs: &R);
}

/// Bounds-checked immutable indexing
// FIXME (AI) `T` should be an associated type
pub trait At<I, T> for Sized? {
    /// Returns an immutable reference to the element at the given index
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the index is out of bounds
    fn at(&self, index: I) -> Result<&T, OutOfBounds>;
}

/// Bounds-checked mutable indexing
// FIXME (AI) `T` should be an associated type
pub trait AtMut<I, T> for Sized? {
    /// Returns a mutable reference to the element at the given index
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the index is out of bounds
    fn at_mut(&mut self, index: I) -> Result<&mut T, OutOfBounds>;
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

/// Linear collection
///
/// **Note** The underlying structure doesn't necessarily store its elements in contiguous memory
pub trait Collection {
    /// Returns the length of the collection
    fn len(&self) -> uint;
}

impl<'a, C> Collection for &'a C where C: Collection {
    fn len(&self) -> uint {
        Collection::len(*self)
    }
}

/// The basic idea of a matrix: A rectangular array arranged in rows and columns
pub trait Matrix {
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

impl<'a, M> Matrix for &'a M where M: Matrix {
    fn ncols(&self) -> uint {
        Matrix::ncols(*self)
    }

    fn nrows(&self) -> uint {
        Matrix::nrows(*self)
    }

    fn size(&self) -> (uint, uint) {
        Matrix::size(*self)
    }
}

/// Immutable view on a column
// FIXME (AI) `'a`, `V` should be associated items
pub trait MatrixCol<'a, V>: Matrix {
    /// Returns an immutable view into the column at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchColumn` if the index is out of bounds
    fn col(&'a self, col: uint) -> ::Result<Col<V>> {
        if col < self.ncols() {
            Ok(unsafe { self.unsafe_col(col) })
        } else {
            Err(Error::NoSuchColumn)
        }
    }

    /// Returns a view into the column at the given index without performing bounds checking
    unsafe fn unsafe_col(&'a self, col: uint) -> Col<V>;
}

/// Mutable access to a column
// FIXME (AI) `'a`, `V` should be associated items
pub trait MatrixColMut<'a, V>: Matrix {
    /// Returns a mutable view into the column at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchColumn` if the index is out of bounds
    fn col_mut(&'a mut self, col: uint) -> ::Result<Col<V>> {
        if col < self.ncols() {
            Ok(unsafe { self.unsafe_col_mut(col) })
        } else {
            Err(Error::NoSuchColumn)
        }
    }

    /// Returns a mutable view into the column at the given index without performing bounds
    /// checking
    unsafe fn unsafe_col_mut(&'a mut self, col: uint) -> Col<V>;
}

/// Immutable column-by-column iteration
// FIXME (AI) `'a` should be an associated lifetime
pub trait MatrixCols<'a>: Matrix {
    /// Returns an iterator that yields immutable views into the columns of the matrix
    fn cols(&'a self) -> Cols<Self> {
        Cols {
            mat: self,
            state: 0,
            stop: self.ncols(),
        }
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
    fn diag(&self, diag: int) -> ::Result<Diag<strided::Slice<T>>>;
}

/// Mutable access to a diagonal
// FIXME (AI) `T` should be an associated type
pub trait MatrixDiagMut<T> {
    /// Returns a mutable view into the diagonal at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchDiagonal` if the index is out of bounds
    fn diag_mut(&mut self, diag: int) -> ::Result<Diag<strided::MutSlice<T>>>;
}

/// Mutable column-by-column iteration
pub trait MatrixMutCols<'a>: Matrix {
    /// Returns an iterator that yields mutable views into the columns of the matrix
    fn mut_cols(&'a mut self) -> MutCols<Self> {
        MutCols {
            stop: self.ncols(),
            mat: self,
            state: 0,
        }
    }
}

/// Mutable row-by-row iteration
pub trait MatrixMutRows<'a>: Matrix {
    /// Returns an iterator that yields mutable views into the rows of the matrix
    fn mut_rows(&'a mut self) -> MutRows<Self> {
        MutRows {
            stop: self.nrows(),
            mat: self,
            state: 0,
        }
    }
}

/// Immutable view into a row
// FIXME (AI) `'a`, `V` should be associated items
pub trait MatrixRow<'a, V>: Matrix {
    /// Returns an immutable view into the row at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchRow` if the index is out of bounds
    fn row(&'a self, row: uint) -> ::Result<Row<V>> {
        if row < self.nrows() {
            Ok(unsafe { self.unsafe_row(row) })
        } else {
            Err(Error::NoSuchRow)
        }
    }

    /// Returns an immutable view into the row at the given index without performing bounds
    /// checking
    unsafe fn unsafe_row(&'a self, row: uint) -> Row<V>;
}

/// Mutable access to a row
// FIXME (AI) `'a`, `V` should be associated items
pub trait MatrixRowMut<'a, V>: Matrix {
    /// Returns a mutable view into the row at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchRow` if the index is out of bounds
    fn row_mut(&'a mut self, row: uint) -> ::Result<Row<V>> {
        if row < self.nrows() {
            Ok(unsafe { self.unsafe_row_mut(row) })
        } else {
            Err(Error::NoSuchRow)
        }
    }

    /// Returns a mutable view into the row at the given index without performing bounds checking
    unsafe fn unsafe_row_mut(&'a mut self, row: uint) -> Row<V>;
}

/// Immutable row-by-row iteration
pub trait MatrixRows<'a>: Matrix {
    /// Returns an iterator that yields immutable views into each row of the matrix
    fn rows(&'a self) -> Rows<'a, Self> {
        Rows {
            mat: self,
            state: 0,
            stop: self.nrows(),
        }
    }
}

/// The `*=` operator
// FIXME (rust-lang/rfcs#393) Use trait provided by the standard library
pub trait MulAssign<R> {
    /// Performs the operation `self *= rhs`
    ///
    /// **Note** The operator sugar has yet to be implemented. See rust-lang/rfcs#393
    fn mul_assign(&mut self, rhs: &R);
}

/// A more flexible slicing trait
///
/// *Note* Sadly this doesn't have operator sugar. You won't be able to use the slicing operator
/// `[]` with this library until Rust gets HKT.
// FIXME (AI) `'a`, `R` should be associated items
pub trait Slice<'a, I, S> for Sized? {
    /// Returns an immutable view into a fraction of the collection that spans `start` : `end`
    fn slice(&'a self, start: I, end: I) -> ::Result<S>;
    /// Convenience method for `slice(start, end_of_collection)`
    fn slice_from(&'a self, start: I) -> ::Result<S>;
    /// Convenience method for `slice(start_of_collection, end)`
    fn slice_to(&'a self, end: I) -> ::Result<S>;
}

/// Mutable version of the `Slice` trait
// FIXME (AI) `'a`, `R` should be associated items
pub trait SliceMut<'a, I, S> for Sized? {
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
    fn sub_assign(&mut self, rhs: &R);
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
