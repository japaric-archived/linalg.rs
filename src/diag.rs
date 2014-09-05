use std::fmt::{Formatter, Show, mod};

use Diag;
use notsafe::{UnsafeIndex, UnsafeIndexMut, UnsafeSlice};
use private::{PrivateIter, PrivateMutIter};
use traits::{Iter, MutIter, OptionIndex, OptionIndexMut, OptionSlice};

impl<D: Collection> Collection for Diag<D> {
    fn len(&self) -> uint {
        self.data.len()
    }
}

impl<T, D: Collection + UnsafeIndex<uint, T>> Index<uint, T> for Diag<D> {
    fn index(&self, &index: &uint) -> &T {
        assert!(index < self.data.len());

        unsafe { self.data.unsafe_index(&index) }
    }
}

impl<T, D: Collection + UnsafeIndexMut<uint, T>> IndexMut<uint, T> for Diag<D> {
    fn index_mut(&mut self, &index: &uint) -> &mut T {
        assert!(index < self.data.len());

        unsafe { self.data.unsafe_index_mut(&index) }
    }
}

impl<T, D: Collection + UnsafeIndex<uint, T>> OptionIndex<uint, T> for Diag<D> {
    fn at(&self, &index: &uint) -> Option<&T> {
        if index < self.data.len() {
            Some(unsafe { self.data.unsafe_index(&index) })
        } else {
            None
        }
    }
}

impl<T, D: Collection + UnsafeIndexMut<uint, T>> OptionIndexMut<uint, T> for Diag<D> {
    fn at_mut(&mut self, &index: &uint) -> Option<&mut T> {
        if index < self.data.len() {
            Some(unsafe { self.data.unsafe_index_mut(&index) })
        } else {
            None
        }
    }
}

impl<'a, T, I: Iterator<T>, D: PrivateIter<'a, T, I>> Iter<'a, T, I> for Diag<D> {
    fn iter(&'a self) -> I {
        self.data.private_iter()
    }
}

impl<'a, T, I: Iterator<T>, D: PrivateMutIter<'a, T, I>> MutIter<'a, T, I> for Diag<D> {
    fn mut_iter(&'a mut self) -> I {
        self.data.private_mut_iter()
    }
}

// TODO Needs testing
impl<'a, D: Collection + UnsafeSlice<'a, uint, D>> OptionSlice<'a, uint, Diag<D>> for Diag<D> {
    fn slice(&'a self, start: uint, end: uint) -> Option<Diag<D>> {
        if end > start + 1 && end <= self.data.len() {
            Some(Diag { data:  unsafe { self.data.unsafe_slice(start, end) }})
        } else {
            None
        }
    }
}

impl<D: Show> Show for Diag<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Diag({})", self.data)
    }
}
