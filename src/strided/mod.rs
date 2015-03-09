//! Strided slices

use std::{fmt, mem};
use std::ops::Range;

use Result;
use error::OutOfBounds;

#[doc(hidden)]
pub struct Slice<'a, T>(pub ::raw::strided::Slice<'a, T>);

impl<'a, T> Slice<'a, T> {
    /// Unreachable
    pub fn at(&self, index: usize) -> ::std::result::Result<&T, OutOfBounds> {
        self.0.at(index)
    }

    /// Unreachable
    pub fn iter(&self) -> Items<T> {
        Items(self.0.iter())
    }

    /// Unreachable
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Unreachable
    pub fn slice(&self, range: Range<usize>) -> Result<Slice<T>> {
        self.0.slice(range).map(Slice)
    }
}

impl<'a, T> Copy for Slice<'a, T> {}

impl<'a, T> ::From<(*const T, usize, usize)> for Slice<'a, T> {
    unsafe fn parts((data, len, stride): (*const T, usize, usize)) -> Slice<'a, T> {
        Slice(::From::parts((data, len, stride)))
    }
}

impl<'a, T> fmt::Debug for Slice<'a, T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

#[doc(hidden)]
pub struct MutSlice<'a, T>(pub ::raw::strided::Slice<'a, T>);

impl<'a, T> MutSlice<'a, T> {
    /// Unreachable
    pub fn at(&self, index: usize) -> ::std::result::Result<&T, OutOfBounds> {
        self.0.at(index)
    }

    /// Unreachable
    pub fn at_mut(&mut self, index: usize) -> ::std::result::Result<&mut T, OutOfBounds> {
        unsafe { mem::transmute(self.0.at(index)) }
    }

    /// Unreachable
    pub fn iter(&self) -> Items<T> {
        Items(self.0.iter())
    }

    /// Unreachable
    pub fn iter_mut(&self) -> MutItems<T> {
        MutItems(self.0.iter())
    }

    /// Unreachable
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Unreachable
    pub fn slice(&self, range: Range<usize>) -> Result<Slice<T>> {
        self.0.slice(range).map(Slice)
    }

    /// Unreachable
    pub fn slice_mut(&mut self, range: Range<usize>) -> Result<MutSlice<T>> {
        self.0.slice(range).map(MutSlice)
    }
}

impl<'a, T> ::From<(*const T, usize, usize)> for MutSlice<'a, T> {
    unsafe fn parts((data, len, stride): (*const T, usize, usize)) -> MutSlice<'a, T> {
        MutSlice(::From::parts((data, len, stride)))
    }
}

impl<'a, T> fmt::Debug for MutSlice<'a, T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

/// Iterator over an immutable strided slice
pub struct Items<'a, T>(::raw::strided::Items<'a, T>);

impl<'a, T> Copy for Items<'a, T> {}

impl<'a, T> DoubleEndedIterator for Items<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        unsafe { mem::transmute(self.0.next_back()) }
    }
}

impl<'a, T> Iterator for Items<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        unsafe { mem::transmute(self.0.next()) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

/// Iterator over an mutable strided slice
pub struct MutItems<'a, T>(::raw::strided::Items<'a, T>);

impl<'a, T> DoubleEndedIterator for MutItems<'a, T> {
    fn next_back(&mut self) -> Option<&'a mut T> {
        unsafe { mem::transmute(self.0.next_back()) }
    }
}

impl<'a, T> Iterator for MutItems<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        unsafe { mem::transmute(self.0.next()) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
