//! Traits

/// A matrix, a rectangular array arranged in rows and columns
pub trait Matrix {
    /// Element of the matrix
    type Elem;

    /// Returns the number of rows of the matrix
    fn nrows(&self) -> u32 {
        self.size().0
    }

    /// Returns the number of columns of the matrix
    fn ncols(&self) -> u32 {
        self.size().1
    }

    /// Returns the size of the matrix
    fn size(&self) -> (u32, u32) {
        (self.nrows(), self.ncols())
    }
}

impl<'a, M: ?Sized> Matrix for &'a M where M: Matrix {
    type Elem = M::Elem;

    fn nrows(&self) -> u32 {
        M::nrows(*self)
    }

    fn ncols(&self) -> u32 {
        M::ncols(*self)
    }

    fn size(&self) -> (u32, u32) {
        M::size(*self)
    }
}

impl<'a, M: ?Sized> Matrix for &'a mut M where M: Matrix {
    type Elem = M::Elem;

    fn nrows(&self) -> u32 {
        M::nrows(*self)
    }

    fn ncols(&self) -> u32 {
        M::ncols(*self)
    }

    fn size(&self) -> (u32, u32) {
        M::size(*self)
    }
}

/// Frobenius norm
pub trait Norm {
    /// The return value
    type Output;

    /// Returns the Frobenius norm of the vector/matrix
    fn norm(&self) -> Self::Output;
}

/// Transpose operator
pub trait Transpose {
    /// Transposed matrix
    type Output;

    /// Returns the transpose of this matrix
    fn t(self) -> Self::Output;
}
