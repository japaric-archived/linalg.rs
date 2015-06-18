//! "Raw" representations

use std::fmt;
use std::marker::PhantomData;

use core::nonzero::NonZero;

use order::Order;
use u31::U31;

/// Raw representation of a strided matrix
pub struct Mat<T, Order> {
    /// Data pointer
    pub data: NonZero<*mut T>,
    /// Number of rows
    pub nrows: U31,
    /// Number of columns
    pub ncols: U31,
    /// Stride
    pub stride: U31,
    /// Marker to "use" the `Order` parameter
    pub marker: PhantomData<Order>,
}

impl<T, O> Clone for Mat<T, O> {
    fn clone(&self) -> Mat<T, O> {
        *self
    }
}

impl<T, O> Copy for Mat<T, O> {}

impl<T, O> fmt::Debug for Mat<T, O> where O: Order {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Mat")
            .field("data", &self.data)
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .field("stride", &self.stride)
            .field("order", &O::order())
            .finish()
    }
}

/// Raw representation of a strided slice
pub struct Slice<T> {
    /// Data pointer
    pub data: NonZero<*mut T>,
    /// Length
    pub len: U31,
    /// Stride
    pub stride: U31,
}

impl<T> Clone for Slice<T> {
    fn clone(&self) -> Slice<T> {
        *self
    }
}

impl<T> Copy for Slice<T> {}

impl<T> fmt::Debug for Slice<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Slice")
            .field("data", &self.data)
            .field("len", &self.len)
            .field("stride", &self.stride)
            .finish()
    }
}
