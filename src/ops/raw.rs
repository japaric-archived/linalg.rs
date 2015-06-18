use std::fmt;

use core::nonzero::NonZero;

use u31::U31;

pub struct Mat<T> {
    pub data: NonZero<*mut T>,
    pub ncols: U31,
    pub nrows: U31,
    pub order: ::Order,
    pub stride: U31,
}

impl<T> Clone for Mat<T> {
    fn clone(&self) -> Mat<T> {
        *self
    }
}

impl<T> Copy for Mat<T> {}

impl<T> fmt::Debug for Mat<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Mat")
            .field("data", &self.data)
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .field("stride", &self.stride)
            .field("order", &self.order)
            .finish()
    }
}
