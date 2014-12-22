//! Strided slices

use std::{fmt, mem};

use Result;
use error::OutOfBounds;

#[doc(hidden)]
pub struct Slice<'a, T: 'a>(pub ::raw::strided::Slice<'a, T>);

impl<'a, T> Slice<'a, T> {
    /// Unreachable
    pub fn at(&self, index: uint) -> ::std::result::Result<&T, OutOfBounds> {
        self.0.at(index)
    }

    /// Unreachable
    pub fn iter(&self) -> Items<T> {
        Items(self.0.iter())
    }

    /// Unreachable
    pub fn len(&self) -> uint {
        self.0.len()
    }

    /// Unreachable
    pub fn slice(&self, start: uint, end: uint) -> Result<Slice<T>> {
        self.0.slice(start, end).map(Slice)
    }
}

impl<'a, T> Copy for Slice<'a, T> {}

impl<'a, T> ::From<(*const T, uint, uint)> for Slice<'a, T> {
    unsafe fn parts((data, len, stride): (*const T, uint, uint)) -> Slice<'a, T> {
        Slice(::From::parts((data, len, stride)))
    }
}

impl<'a, T> fmt::Show for Slice<'a, T> where T: fmt::Show {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Show::fmt(&self.0, f)
    }
}

#[doc(hidden)]
pub struct MutSlice<'a, T: 'a>(pub ::raw::strided::Slice<'a, T>);

impl<'a, T> MutSlice<'a, T> {
    /// Unreachable
    pub fn at(&self, index: uint) -> ::std::result::Result<&T, OutOfBounds> {
        self.0.at(index)
    }

    /// Unreachable
    pub fn at_mut(&mut self, index: uint) -> ::std::result::Result<&mut T, OutOfBounds> {
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
    pub fn len(&self) -> uint {
        self.0.len()
    }

    /// Unreachable
    pub fn slice(&self, start: uint, end: uint) -> Result<Slice<T>> {
        self.0.slice(start, end).map(Slice)
    }

    /// Unreachable
    pub fn slice_mut(&mut self, start: uint, end: uint) -> Result<MutSlice<T>> {
        self.0.slice(start, end).map(MutSlice)
    }
}

impl<'a, T> ::From<(*const T, uint, uint)> for MutSlice<'a, T> {
    unsafe fn parts((data, len, stride): (*const T, uint, uint)) -> MutSlice<'a, T> {
        MutSlice(::From::parts((data, len, stride)))
    }
}

impl<'a, T> fmt::Show for MutSlice<'a, T> where T: fmt::Show {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Show::fmt(&self.0, f)
    }
}

/// Iterator over an immutable strided slice
pub struct Items<'a, T: 'a>(::raw::strided::Items<'a, T>);

impl<'a, T> Copy for Items<'a, T> {}

impl<'a, T> DoubleEndedIterator<&'a T> for Items<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        unsafe { mem::transmute(self.0.next_back()) }
    }
}

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        unsafe { mem::transmute(self.0.next()) }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.0.size_hint()
    }
}

/// Iterator over an mutable strided slice
pub struct MutItems<'a, T: 'a>(::raw::strided::Items<'a, T>);

impl<'a, T> DoubleEndedIterator<&'a mut T> for MutItems<'a, T> {
    fn next_back(&mut self) -> Option<&'a mut T> {
        unsafe { mem::transmute(self.0.next_back()) }
    }
}

impl<'a, T> Iterator<&'a mut T> for MutItems<'a, T> {
    fn next(&mut self) -> Option<&'a mut T> {
        unsafe { mem::transmute(self.0.next()) }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.0.size_hint()
    }
}
