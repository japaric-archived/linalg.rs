use std::fmt::{Formatter, Show, mod};

use Col;
use blas::copy::BlasCopy;
use notsafe::{UnsafeIndex, UnsafeIndexMut, UnsafeSlice};
use private::{PrivateIter, PrivateMutIter, PrivateToOwned};
use traits::{Iter, Matrix, MutIter, OptionIndex, OptionIndexMut, OptionSlice, ToOwned};

impl<D: Collection> Collection for Col<D> {
    fn len(&self) -> uint {
        self.data.len()
    }
}

impl<T, D: Collection + UnsafeIndex<uint, T>> Index<uint, T> for Col<D> {
    fn index(&self, &index: &uint) -> &T {
        assert!(index < self.len());

        unsafe { self.data.unsafe_index(&index) }
    }
}

impl<T, D: Collection + UnsafeIndexMut<uint, T>> IndexMut<uint, T> for Col<D> {
    fn index_mut(&mut self, &index: &uint) -> &mut T {
        assert!(index < self.len());

        unsafe { self.data.unsafe_index_mut(&index) }
    }
}

impl<T, D: Collection + UnsafeIndex<uint, T>> OptionIndex<uint, T> for Col<D> {
    fn at(&self, &index: &uint) -> Option<&T> {
        if index < self.len() {
            Some(unsafe { self.data.unsafe_index(&index) })
        } else {
            None
        }
    }
}

impl<T, D: Collection + UnsafeIndexMut<uint, T>> OptionIndexMut<uint, T> for Col<D> {
    fn at_mut(&mut self, &index: &uint) -> Option<&mut T> {
        if index < self.len() {
            Some(unsafe { self.data.unsafe_index_mut(&index) })
        } else {
            None
        }
    }
}

impl<'a, T, I: Iterator<T>, D: PrivateIter<'a, T, I>> Iter<'a, T, I> for Col<D> {
    fn iter(&'a self) -> I {
        self.data.private_iter()
    }
}

impl<D: Collection> Matrix for Col<D> {
    fn ncols(&self) -> uint {
        1
    }

    fn nrows(&self) -> uint {
        self.len()
    }
}

impl<'a, T, I: Iterator<T>, D: PrivateMutIter<'a, T, I>> MutIter<'a, T, I> for Col<D> {
    fn mut_iter(&'a mut self) -> I {
        self.data.private_mut_iter()
    }
}

// TODO Needs testing
impl<'a, D: Collection + UnsafeSlice<'a, uint, D>> OptionSlice<'a, uint, Col<D>> for Col<D> {
    fn slice(&'a self, start: uint, end: uint) -> Option<Col<D>> {
        if end > start + 1 && end <= self.data.len() {
            Some(Col { data:  unsafe { self.data.unsafe_slice(start, end) }})
        } else {
            None
        }
    }
}

impl<T: BlasCopy, D: PrivateToOwned<T>> ToOwned<Col<Vec<T>>> for Col<D> {
    fn to_owned(&self) -> Col<Vec<T>> {
        Col {
            data: self.data.private_to_owned(),
        }
    }
}

impl<D: Show> Show for Col<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Col({})", self.data)
    }
}
