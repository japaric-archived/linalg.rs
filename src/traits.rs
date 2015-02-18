//! Traits

use error::OutOfBounds;
use {Col, Cols, Diag, Error, MutCol, MutCols, MutDiag, MutRow, MutRows, Result, Row, Rows};

/// Bounds-checked immutable indexing
pub trait At<I> {
    type Output;

    /// Returns an immutable reference to the element at the given index
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the index is out of bounds
    fn at(&self, index: I) -> ::std::result::Result<&Self::Output, OutOfBounds>;
}

/// Bounds-checked mutable indexing
pub trait AtMut<I> {
    type Output;

    /// Returns a mutable reference to the element at the given index
    ///
    /// # Errors
    ///
    /// - `OutOfBounds` if the index is out of bounds
    fn at_mut(&mut self, index: I) -> ::std::result::Result<&mut Self::Output, OutOfBounds>;
}

/// Immutable iteration over a collection
// FIXME (AI) `'a` should be associated items
pub trait Iter<'a> {
    type Iter: Iterator;

    /// Returns an iterator that yields immutable references to the elements of the collection
    fn iter(&'a self) -> Self::Iter;
}

/// Mutable iteration over a collection
// FIXME (AI) `'a`, should be associated items
pub trait IterMut<'a> {
    type Iter: Iterator;

    /// Returns an iterator that yields mutable references to the elements of the collection
    fn iter_mut(&'a mut self) -> Self::Iter;
}

/// The basic idea of a matrix: A rectangular array arranged in rows and columns
pub trait Matrix: Sized {
    type Elem;

    /// Returns the number of columns the matrix has
    fn ncols(&self) -> usize {
        self.size().1
    }

    /// Returns the number of rows the matrix has
    fn nrows(&self) -> usize {
        self.size().0
    }

    /// Returns the size of the matrix
    fn size(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
}

/// Immutable view on a column
pub trait MatrixCol: Matrix {
    /// Returns an immutable view into the column at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchColumn` if the index is out of bounds
    fn col(&self, col: usize) -> Result<Col<Self::Elem>> {
        if col < self.ncols() {
            Ok(unsafe { self.unsafe_col(col) })
        } else {
            Err(Error::NoSuchColumn)
        }
    }

    /// Returns a view into the column at the given index without performing bounds checking
    unsafe fn unsafe_col(&self, col: usize) -> Col<Self::Elem>;
}

/// Mutable access to a column
pub trait MatrixColMut: MatrixCol {
    /// Returns a mutable view into the column at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchColumn` if the index is out of bounds
    fn col_mut(&mut self, col: usize) -> Result<MutCol<Self::Elem>> {
        if col < self.ncols() {
            Ok(unsafe { self.unsafe_col_mut(col) })
        } else {
            Err(Error::NoSuchColumn)
        }
    }

    /// Returns a mutable view into the column at the given index without performing bounds
    /// checking
    unsafe fn unsafe_col_mut(&mut self, col: usize) -> MutCol<Self::Elem>;
}

/// Immutable column-by-column iteration
pub trait MatrixCols: Matrix {
    /// Returns an iterator that yields immutable views into the columns of the matrix
    fn cols(&self) -> Cols<Self> {
        Cols(unsafe { ::From::parts(self) })
    }
}

/// Immutable view on a diagonal
pub trait MatrixDiag: Matrix {
    /// Returns a view into the diagonal at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchDiagonal` if the index is out of bounds
    fn diag(&self, diag: isize) -> Result<Diag<Self::Elem>>;
}

/// Mutable access to a diagonal
pub trait MatrixDiagMut: Matrix {
    /// Returns a mutable view into the diagonal at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchDiagonal` if the index is out of bounds
    fn diag_mut(&mut self, diag: isize) -> ::Result<MutDiag<Self::Elem>>;
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
pub trait MatrixRow: Matrix {
    /// Returns an immutable view into the row at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchRow` if the index is out of bounds
    fn row(&self, row: usize) -> Result<Row<Self::Elem>> {
        if row < self.nrows() {
            Ok(unsafe { self.unsafe_row(row) })
        } else {
            Err(Error::NoSuchRow)
        }
    }

    /// Returns an immutable view into the row at the given index without performing bounds
    /// checking
    unsafe fn unsafe_row(&self, row: usize) -> Row<Self::Elem>;
}

/// Immutable row-by-row iteration
pub trait MatrixRows: Matrix {
    /// Returns an iterator that yields immutable views into each row of the matrix
    fn rows(&self) -> Rows<Self> {
        Rows(unsafe { ::From::parts(self) })
    }
}

/// Mutable access to a row
pub trait MatrixRowMut: MatrixRow {
    /// Returns a mutable view into the row at the given index
    ///
    /// # Errors
    ///
    /// - `NoSuchRow` if the index is out of bounds
    fn row_mut(&mut self, row: usize) -> Result<MutRow<Self::Elem>> {
        if row < self.nrows() {
            Ok(unsafe { self.unsafe_row_mut(row) })
        } else {
            Err(Error::NoSuchRow)
        }
    }

    /// Returns a mutable view into the row at the given index without performing bounds checking
    unsafe fn unsafe_row_mut(&mut self, row: usize) -> MutRow<Self::Elem>;
}

/// A more flexible slicing trait
///
/// *Note* Sadly this doesn't have operator sugar. You won't be able to use the slicing operator
/// `[]` with this library until Rust gets HKT.
// FIXME (AI) `'a` should be associated items
pub trait Slice<'a, I> {
    type Slice;

    /// Returns an immutable view into a fraction of the collection that spans `start` : `end`
    fn slice(&'a self, start: I, end: I) -> ::Result<Self::Slice>;
    /// Convenience method for `slice(start, end_of_collection)`
    fn slice_from(&'a self, start: I) -> ::Result<Self::Slice>;
    /// Convenience method for `slice(start_of_collection, end)`
    fn slice_to(&'a self, end: I) -> ::Result<Self::Slice>;
}

/// Mutable version of the `Slice` trait
// FIXME (AI) `'a`, should be associated items
pub trait SliceMut<'a, I> {
    type Slice;

    /// Returns a mutable view into a fraction of the collection that spans `start` : `end`
    fn slice_mut(&'a mut self, start: I, end: I) -> ::Result<Self::Slice>;
    /// Convenience method for `slice_mut(start, end_of_collection)`
    fn slice_from_mut(&'a mut self, start: I) -> ::Result<Self::Slice>;
    /// Convenience method for `slice_mut(start_of_collection, end)`
    fn slice_to_mut(&'a mut self, end: I) -> ::Result<Self::Slice>;
}

/// Make an owned clone from a view
// TODO (rust-lang/rust#18910) Use trait provided by the standard library
pub trait ToOwned<T> {
    /// Returns an owned clone from the view
    fn to_owned(&self) -> T;
}

/// The transpose operator
pub trait Transpose {
    type Output;

    /// Returns the transpose of the input
    fn t(self) -> Self::Output;
}
