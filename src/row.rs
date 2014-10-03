use std::fmt::{Formatter, Show, mod};

use blas::{BlasPtr, BlasStride, to_blasint};
use blas::copy::BlasCopy;
use blas::dot::BlasDot;
use notsafe::{UnsafeIndex, UnsafeIndexMut, UnsafeSlice};
use private::{PrivateIter, PrivateMutIter, PrivateToOwned};
use traits::{Iter, Matrix, MutIter, OptionIndex, OptionIndexMut, OptionSlice, ToOwned};
use {Col, Row};

impl<D> Collection for Row<D> where D: Collection {
    fn len(&self) -> uint {
        self.0.len()
    }
}

impl<T, D> Index<uint, T> for Row<D> where D: Collection + UnsafeIndex<uint, T> {
    fn index(&self, &index: &uint) -> &T {
        assert!(index < self.len());

        unsafe { self.0.unsafe_index(&index) }
    }
}

impl<T, D> IndexMut<uint, T> for Row<D> where D: Collection + UnsafeIndexMut<uint, T> {
    fn index_mut(&mut self, &index: &uint) -> &mut T {
        assert!(index < self.len());

        unsafe { self.0.unsafe_index_mut(&index) }
    }
}

impl<T, D> OptionIndex<uint, T> for Row<D> where D: Collection + UnsafeIndex<uint, T> {
    fn at(&self, &index: &uint) -> Option<&T> {
        if index < self.len() {
            Some(unsafe { self.0.unsafe_index(&index) })
        } else {
            None
        }
    }
}

impl<T, D> OptionIndexMut<uint, T> for Row<D> where D: Collection + UnsafeIndexMut<uint, T> {
    fn at_mut(&mut self, &index: &uint) -> Option<&mut T> {
        if index < self.len() {
            Some(unsafe { self.0.unsafe_index_mut(&index) })
        } else {
            None
        }
    }
}

impl<'a, T, I, D> Iter<'a, T, I> for Row<D> where I: Iterator<T>, D: PrivateIter<'a, T, I> {
    fn iter(&'a self) -> I {
        self.0.private_iter()
    }
}

impl<D> Matrix for Row<D> where D: Collection {
    fn ncols(&self) -> uint {
        self.len()
    }

    fn nrows(&self) -> uint {
        1
    }
}

impl<T, D, E> Mul<Col<E>, T> for Row<D> where
    T: BlasDot,
    D: BlasPtr<T> + BlasStride + Collection,
    E: BlasPtr<T> + BlasStride + Collection,
{
    /// - Time: `O(length)`
    ///
    /// # Failure
    ///
    /// Fails if the length of the vectors are different
    fn mul(&self, rhs: &Col<E>) -> T {
        assert!(self.len() == rhs.len());

        let dot = BlasDot::dot(None::<T>);
        let n = &to_blasint(self.len());
        let x = self.0.blas_ptr();
        let incx = &self.0.blas_stride();
        let y = rhs.0.blas_ptr();
        let incy = &rhs.0.blas_stride();

        unsafe { dot(n, x, incx, y, incy) }
    }
}

impl<'a, T, I, D> MutIter<'a, T, I> for Row<D> where
    I: Iterator<T>,
    D: PrivateMutIter<'a, T, I>
{
    fn mut_iter(&'a mut self) -> I {
        self.0.private_mut_iter()
    }
}

// TODO Needs testing
impl<'a, D> OptionSlice<'a, uint, Row<D>> for Row<D> where
    D: Collection + UnsafeSlice<'a, uint, D>
{
    fn slice(&'a self, start: uint, end: uint) -> Option<Row<D>> {
        if end > start + 1 && end <= self.0.len() {
            Some(Row(unsafe { self.0.unsafe_slice(start, end) }))
        } else {
            None
        }
    }
}

impl<T, D> ToOwned<Row<Vec<T>>> for Row<D> where T: BlasCopy, D: PrivateToOwned<T> {
    fn to_owned(&self) -> Row<Vec<T>> {
        Row(self.0.private_to_owned())
    }
}

impl<D> Show for Row<D> where D: Show {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Row({})", self.0)
    }
}
