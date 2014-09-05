use std::fmt::{Formatter, Show, mod};

use blas::{BlasPtr, BlasStride, to_blasint};
use blas::copy::BlasCopy;
use blas::dot::BlasDot;
use notsafe::{UnsafeIndex, UnsafeIndexMut, UnsafeSlice};
use private::{PrivateIter, PrivateMutIter, PrivateToOwned};
use traits::{Iter, Matrix, MutIter, OptionIndex, OptionIndexMut, OptionSlice, ToOwned};
use {Col, Row};

impl<D: Collection> Collection for Row<D> {
    fn len(&self) -> uint {
        self.data.len()
    }
}

impl<T, D: Collection + UnsafeIndex<uint, T>> Index<uint, T> for Row<D> {
    fn index(&self, &index: &uint) -> &T {
        assert!(index < self.len());

        unsafe { self.data.unsafe_index(&index) }
    }
}

impl<T, D: Collection + UnsafeIndexMut<uint, T>> IndexMut<uint, T> for Row<D> {
    fn index_mut(&mut self, &index: &uint) -> &mut T {
        assert!(index < self.len());

        unsafe { self.data.unsafe_index_mut(&index) }
    }
}

impl<T, D: Collection + UnsafeIndex<uint, T>> OptionIndex<uint, T> for Row<D> {
    fn at(&self, &index: &uint) -> Option<&T> {
        if index < self.len() {
            Some(unsafe { self.data.unsafe_index(&index) })
        } else {
            None
        }
    }
}

impl<T, D: Collection + UnsafeIndexMut<uint, T>> OptionIndexMut<uint, T> for Row<D> {
    fn at_mut(&mut self, &index: &uint) -> Option<&mut T> {
        if index < self.len() {
            Some(unsafe { self.data.unsafe_index_mut(&index) })
        } else {
            None
        }
    }
}

impl<'a, T, I: Iterator<T>, D: PrivateIter<'a, T, I>> Iter<'a, T, I> for Row<D> {
    fn iter(&'a self) -> I {
        self.data.private_iter()
    }
}

impl<D: Collection> Matrix for Row<D> {
    fn ncols(&self) -> uint {
        self.len()
    }

    fn nrows(&self) -> uint {
        1
    }
}

impl<
    T: BlasDot,
    D: BlasPtr<T> + BlasStride + Collection,
    E: BlasPtr<T> + BlasStride + Collection,
> Mul<Col<E>, T> for Row<D> {
    /// - Time: `O(length)`
    ///
    /// # Failure
    ///
    /// Fails if the length of the vectors are different
    fn mul(&self, rhs: &Col<E>) -> T {
        assert!(self.len() == rhs.len());

        let dot = BlasDot::dot(None::<T>);
        let n = &to_blasint(self.len());
        let x = self.data.blas_ptr();
        let incx = &self.data.blas_stride();
        let y = rhs.data.blas_ptr();
        let incy = &rhs.data.blas_stride();

        unsafe { dot(n, x, incx, y, incy) }
    }
}

impl<'a, T, I: Iterator<T>, D: PrivateMutIter<'a, T, I>> MutIter<'a, T, I> for Row<D> {
    fn mut_iter(&'a mut self) -> I {
        self.data.private_mut_iter()
    }
}

// TODO Needs testing
impl<'a, D: Collection + UnsafeSlice<'a, uint, D>> OptionSlice<'a, uint, Row<D>> for Row<D> {
    fn slice(&'a self, start: uint, end: uint) -> Option<Row<D>> {
        if end > start + 1 && end <= self.data.len() {
            Some(Row { data: unsafe { self.data.unsafe_slice(start, end) }})
        } else {
            None
        }
    }
}

impl<T: BlasCopy, D: PrivateToOwned<T>> ToOwned<Row<Vec<T>>> for Row<D> {
    fn to_owned(&self) -> Row<Vec<T>> {
        Row {
            data: self.data.private_to_owned(),
        }
    }
}

impl<D: Show> Show for Row<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Row({})", self.data)
    }
}
