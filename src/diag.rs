use std::fmt::{Formatter, Show, mod};

use Diag;
use notsafe::{UnsafeIndex, UnsafeIndexMut, UnsafeSlice};
use private::{PrivateIter, PrivateMutIter};
use traits::{Collection, Iter, MutIter, OptionIndex, OptionIndexMut, OptionSlice};

impl<D> Collection for Diag<D> where D: Collection {
    fn len(&self) -> uint {
        self.0.len()
    }
}

impl<T, D> Index<uint, T> for Diag<D> where D: Collection + UnsafeIndex<uint, T> {
    fn index(&self, &index: &uint) -> &T {
        assert!(index < self.0.len());

        unsafe { self.0.unsafe_index(&index) }
    }
}

impl<T, D> IndexMut<uint, T> for Diag<D> where D: Collection + UnsafeIndexMut<uint, T> {
    fn index_mut(&mut self, &index: &uint) -> &mut T {
        assert!(index < self.0.len());

        unsafe { self.0.unsafe_index_mut(&index) }
    }
}

impl<T, D> OptionIndex<uint, T> for Diag<D> where D: Collection + UnsafeIndex<uint, T> {
    fn at(&self, &index: &uint) -> Option<&T> {
        if index < self.0.len() {
            Some(unsafe { self.0.unsafe_index(&index) })
        } else {
            None
        }
    }
}

impl<T, D> OptionIndexMut<uint, T> for Diag<D> where D: Collection + UnsafeIndexMut<uint, T> {
    fn at_mut(&mut self, &index: &uint) -> Option<&mut T> {
        if index < self.0.len() {
            Some(unsafe { self.0.unsafe_index_mut(&index) })
        } else {
            None
        }
    }
}

impl<'a, T, I, D> Iter<'a, T, I> for Diag<D> where I: Iterator<T>, D: PrivateIter<'a, T, I> {
    fn iter(&'a self) -> I {
        self.0.private_iter()
    }
}

impl<'a, T, I, D> MutIter<'a, T, I> for Diag<D> where I: Iterator<T>, D: PrivateMutIter<'a, T, I> {
    fn mut_iter(&'a mut self) -> I {
        self.0.private_mut_iter()
    }
}

// TODO Needs testing
impl<'a, D> OptionSlice<'a, uint, Diag<D>> for Diag<D> where
    D: Collection + UnsafeSlice<'a, uint, D>
{
    fn slice(&'a self, start: uint, end: uint) -> Option<Diag<D>> {
        if end > start + 1 && end <= self.0.len() {
            Some(Diag(unsafe { self.0.unsafe_slice(start, end) }))
        } else {
            None
        }
    }
}

impl<D> Show for Diag<D> where D: Show {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Diag({})", self.0)
    }
}
