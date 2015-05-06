//! Common implementations for linear collections

use std::iter::{FromIterator, IntoIterator};
use std::ops::{Index, IndexMut, Range, RangeFull, RangeFrom, RangeTo};
use std::slice;

use cast::From as _0;
use onezero::{One, Zero};

use strided;
use traits::{Matrix, Slice, SliceMut, Transpose};
use {Col, ColMut, ColVec, Diag, DiagMut, Row, RowMut, RowVec, Tor};

impl<'a, T> Col<'a, T> {
    /// Returns a slice that contains the whole vector
    pub fn as_slice(&self) -> Option<&[T]> {
        self.0.as_slice()
    }

    /// Returns an "immutable iterator" over the column
    pub fn iter(&self) -> strided::Iter<'a, T> {
        self.0.iter()
    }

    fn len(&self) -> u32 {
        self.0.len()
    }
}

impl<'a, T> ColMut<'a, T> {
    /// Returns a slice that contains the whole vector
    pub fn as_slice(&self) -> Option<&[T]> {
        self.0.as_slice()
    }

    /// Returns a mutable slice that contains the whole vector
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        (self.0).0.as_slice_mut()
    }

    /// Returns an "immutable iterator" over the column
    pub fn iter(&self) -> strided::Iter<T> {
        self.0.iter()
    }

    fn len(&self) -> u32 {
        self.0.len()
    }
}

impl<T> ColVec<T> {
    /// Creates a column vector from an owned slice
    ///
    /// # Panics
    ///
    /// If `elems.len() > 2^31`
    pub fn new(elems: Box<[T]>) -> ColVec<T> {
        ColVec::from(elems)
    }

    /// Creates a column vector of size `length` filled with ones
    ///
    /// # Panics
    ///
    /// If `length > 2^31`
    pub fn ones(length: u32) -> ColVec<T> where T: Clone + One {
        unsafe {
            ColVec(Tor::ones(i32::from_(length).unwrap()))
        }
    }

    /// Creates a column vector of size `length` filled with zeros
    ///
    /// # Panics
    ///
    /// If `length > 2^31`
    pub fn zeros(length: u32) -> ColVec<T> where T: Clone + Zero {
        unsafe {
            ColVec(Tor::zeros(i32::from_(length).unwrap()))
        }
    }

    /// Returns a slice that contains the whole vector
    pub fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }

    /// Returns a mutable slice that contains the whole vector
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.0.as_slice_mut()
    }

    /// Returns an "immutable iterator" over the vector
    pub fn iter(&self) -> slice::Iter<T> {
        self.0.iter()
    }

    /// Returns a "mutable iterator" over the vector
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.0.iter_mut()
    }

    fn len(&self) -> u32 {
        self.0.len()
    }
}

impl<'a, T> Diag<'a, T> {
    /// Returns an "immutable iterator" over the diagonal
    pub fn iter(&self) -> strided::Iter<'a, T> {
        self.0.iter()
    }

    /// Returns the length of the diagonal
    pub fn len(&self) -> u32 {
        self.0.len()
    }
}

impl<'a, T> DiagMut<'a, T> {
    /// Returns an "immutable iterator" over the diagonal
    pub fn iter(&self) -> strided::Iter<T> {
        self.0.iter()
    }

    /// Returns the length of the diagonal
    pub fn len(&self) -> u32 {
        self.0.len()
    }
}

impl<'a, T> Row<'a, T> {
    /// Returns a slice that contains the whole vector
    pub fn as_slice(&self) -> Option<&[T]> {
        self.0.as_slice()
    }

    /// Returns an "immutable iterator" over the row
    pub fn iter(&self) -> strided::Iter<'a, T> {
        self.0.iter()
    }

    fn len(&self) -> u32 {
        self.0.len()
    }
}

impl<'a, T> RowMut<'a, T> {
    /// Returns a slice that contains the whole vector
    pub fn as_slice(&self) -> Option<&[T]> {
        self.0.as_slice()
    }

    /// Returns a mutable slice that contains the whole vector
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        (self.0).0.as_slice_mut()
    }

    /// Returns an "immutable iterator" over the row
    pub fn iter(&self) -> strided::Iter<T> {
        self.0.iter()
    }

    fn len(&self) -> u32 {
        self.0.len()
    }
}

impl<T> RowVec<T> {
    /// Creates an owned row vector from an owned slice
    ///
    /// # Panics
    ///
    /// If `elems.len() > 2^31`
    pub fn new(elems: Box<[T]>) -> RowVec<T> {
        RowVec::from(elems)
    }

    /// Returns a slice that contains the whole vector
    pub fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }

    /// Returns a mutable slice that contains the whole vector
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.0.as_slice_mut()
    }

    /// Creates a row vector of size `length` filled with ones
    ///
    /// # Panics
    ///
    /// If `length > 2^31`
    pub fn ones(length: u32) -> RowVec<T> where T: Clone + One {
        unsafe {
            RowVec(Tor::ones(i32::from_(length).unwrap()))
        }
    }

    /// Creates a row vector of size `length` filled with zeros
    ///
    /// # Panics
    ///
    /// If `length > 2^31`
    pub fn zeros(length: u32) -> RowVec<T> where T: Clone + Zero {
        unsafe {
            RowVec(Tor::zeros(i32::from_(length).unwrap()))
        }
    }

    /// Returns an "immutable iterator" over the vector
    pub fn iter(&self) -> slice::Iter<T> {
        self.0.iter()
    }

    /// Returns a "mutable iterator" over the vector
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.0.iter_mut()
    }

    fn len(&self) -> u32 {
        self.0.len()
    }
}

macro_rules! index {
    ($ty:ident, $ty_mut:ident, $ty_owned:ident) => {
        index!($ty, $ty_mut);

        impl<T> Index<u32> for $ty_owned<T> {
            type Output = T;

            fn index(&self, i: u32) -> &T {
                unsafe {
                    &*self.0.raw_index(i)
                }
            }
        }

        impl<T> IndexMut<u32> for $ty_owned<T> {
            fn index_mut(&mut self, i: u32) -> &mut T {
                unsafe {
                    &mut *self.0.raw_index(i)
                }
            }
        }
    };
    ($ty:ident, $ty_mut:ident) => {
        impl<'a, T> Index<u32> for $ty<'a, T> {
            type Output = T;

            fn index(&self, i: u32) -> &T {
                unsafe {
                    &*self.0.raw_index(i)
                }
            }
        }

        impl<'a, T> IndexMut<u32> for $ty_mut<'a, T> {
            fn index_mut(&mut self, i: u32) -> &mut T {
                unsafe {
                    &mut *(self.0).0.raw_index(i)
                }
            }
        }

        impl<'a, T> Index<u32> for $ty_mut<'a, T> {
            type Output = T;

            fn index(&self, i: u32) -> &T {
                &self.0[i]
            }
        }
    };
}

index!(Col, ColMut, ColVec);
index!(Diag, DiagMut);
index!(Row, RowMut, RowVec);

macro_rules! from_iterator {
    ($($ty:ident),+) => {
        $(
            /// # Panics
            ///
            /// If the iterator yields more than `2^31` elements
            impl<T> FromIterator<T> for $ty<T> {
                fn from_iter<I>(it: I) -> $ty<T> where I: IntoIterator<Item=T> {
                    $ty(it.into_iter().collect::<Tor<_>>())
                }
            }
         )+
    };
}

from_iterator!(ColVec, RowVec);

// IntoIterator
macro_rules! into_iter {
    ($ty:ident, $ty_mut:ident, $ty_owned:ident) => {
        impl<'a, T> IntoIterator for &'a $ty_owned<T> {
            type IntoIter = slice::Iter<'a, T>;
            type Item = &'a T;

            fn into_iter(self) -> slice::Iter<'a, T> {
                self.iter()
            }
        }

        impl<'a, T> IntoIterator for &'a mut $ty_owned<T> {
            type IntoIter = slice::IterMut<'a, T>;
            type Item = &'a mut T;

            fn into_iter(self) -> slice::IterMut<'a, T> {
                self.iter_mut()
            }
        }

        into_iter!($ty, $ty_mut);
    };
    ($ty:ident, $ty_mut:ident) => {
        impl<'a, T> IntoIterator for $ty<'a, T> {
            type IntoIter = strided::Iter<'a, T>;
            type Item = &'a T;

            fn into_iter(self) -> strided::Iter<'a, T> {
                self.0.iter()
            }
        }

        impl<'a, 'b, T> IntoIterator for &'a $ty_mut<'b, T> {
            type IntoIter = strided::Iter<'a, T>;
            type Item = &'a T;

            fn into_iter(self) -> strided::Iter<'a, T> {
                self.iter()
            }
        }

        impl<'a, 'b, T> IntoIterator for &'a mut $ty_mut<'b, T> {
            type IntoIter = strided::IterMut<'a, T>;
            type Item = &'a mut T;

            fn into_iter(self) -> strided::IterMut<'a, T> {
                self.iter_mut()
            }
        }
    };
}

into_iter!(Col, ColMut, ColVec);
into_iter!(Diag, DiagMut);
into_iter!(Row, RowMut, RowVec);

impl<'a, T> Matrix for Col<'a, T> {
    type Elem = T;

    fn ncols(&self) -> u32 {
        1
    }

    fn nrows(&self) -> u32 {
        self.len()
    }
}

impl<'a, T> Matrix for ColMut<'a, T> {
    type Elem = T;

    fn ncols(&self) -> u32 {
        1
    }

    fn nrows(&self) -> u32 {
        self.len()
    }
}

impl<T> Matrix for ColVec<T> {
    type Elem = T;

    fn ncols(&self) -> u32 {
        1
    }

    fn nrows(&self) -> u32 {
        self.len()
    }
}

impl<'a, T> Matrix for Row<'a, T> {
    type Elem = T;

    fn ncols(&self) -> u32 {
        self.len()
    }

    fn nrows(&self) -> u32 {
        1
    }
}

impl<'a, T> Matrix for RowMut<'a, T> {
    type Elem = T;

    fn ncols(&self) -> u32 {
        self.len()
    }

    fn nrows(&self) -> u32 {
        1
    }
}

impl<T> Matrix for RowVec<T> {
    type Elem = T;

    fn ncols(&self) -> u32 {
        self.len()
    }

    fn nrows(&self) -> u32 {
        1
    }
}

macro_rules! slice {
    ($ty_owned:ident, $ty:ident, $ty_mut:ident) => {
        impl<'a, T> Slice<'a, RangeFull> for $ty_owned<T> {
            type Output = $ty<'a, T>;

            fn slice(&'a self, _: RangeFull) -> $ty<'a, T> {
                unsafe {
                    use Slice;

                    $ty(Slice::new(*self.0.data, self.0.len, 1))
                }
            }
        }

        impl<'a, T> Slice<'a, Range<u32>> for $ty_owned<T> {
            type Output = $ty<'a, T>;

            fn slice(&'a self, r: Range<u32>) -> $ty<'a, T> {
                unsafe {
                    let v: *const $ty<T> = &self.slice(..);
                    (*v).slice(r)
                }
            }
        }

        impl<'a, T> Slice<'a, RangeFrom<u32>> for $ty_owned<T> {
            type Output = $ty<'a, T>;

            fn slice(&'a self, r: RangeFrom<u32>) -> $ty<'a, T> {
                self.slice(r.start..self.len())
            }
        }

        impl<'a, T> Slice<'a, RangeTo<u32>> for $ty_owned<T> {
            type Output = $ty<'a, T>;

            fn slice(&'a self, r: RangeTo<u32>) -> $ty<'a, T> {
                self.slice(0..r.end)
            }
        }

        impl<'a, T> SliceMut<'a, RangeFull> for $ty_owned<T> {
            type Output = $ty_mut<'a, T>;

            fn slice_mut(&'a mut self, _: RangeFull) -> $ty_mut<'a, T> {
                $ty_mut(self.slice(..))
            }
        }

        impl<'a, T> SliceMut<'a, Range<u32>> for $ty_owned<T> {
            type Output = $ty_mut<'a, T>;

            fn slice_mut(&'a mut self, r: Range<u32>) -> $ty_mut<'a, T> {
                $ty_mut(self.slice(r))
            }
        }

        impl<'a, T> SliceMut<'a, RangeFrom<u32>> for $ty_owned<T> {
            type Output = $ty_mut<'a, T>;

            fn slice_mut(&'a mut self, r: RangeFrom<u32>) -> $ty_mut<'a, T> {
                $ty_mut(self.slice(r.start..self.len()))
            }
        }

        impl<'a, T> SliceMut<'a, RangeTo<u32>> for $ty_owned<T> {
            type Output = $ty_mut<'a, T>;

            fn slice_mut(&'a mut self, r: RangeTo<u32>) -> $ty_mut<'a, T> {
                $ty_mut(self.slice(0..r.end))
            }
        }

        slice!($ty, $ty_mut);
    };
    ($ty:ident, $ty_mut:ident) => {
        impl<'a, 'b, T> Slice<'a, Range<u32>> for $ty<'b, T> {
            type Output = $ty<'b, T>;

            fn slice(&'a self, r: Range<u32>) -> $ty<'b, T> {
                $ty(self.0.slice(r))
            }
        }

        impl<'a, 'b, T> Slice<'a, RangeFrom<u32>> for $ty<'b, T> {
            type Output = $ty<'b, T>;

            fn slice(&'a self, r: RangeFrom<u32>) -> $ty<'b, T> {
                $ty(self.0.slice(r.start..self.len()))
            }
        }

        impl<'a, 'b, T> Slice<'a, RangeFull> for $ty<'b, T> {
            type Output = $ty<'b, T>;

            fn slice(&'a self, _: RangeFull) -> $ty<'b, T> {
                *self
            }
        }

        impl<'a, 'b, T> Slice<'a, RangeTo<u32>> for $ty<'b, T> {
            type Output = $ty<'b, T>;

            fn slice(&'a self, r: RangeTo<u32>) -> $ty<'b, T> {
                $ty(self.0.slice(0..r.end))
            }
        }

        impl<'a, 'b, T> Slice<'a, Range<u32>> for $ty_mut<'b, T> {
            type Output = $ty<'a, T>;

            fn slice(&'a self, r: Range<u32>) -> $ty<'a, T> {
                self.0.slice(r)
            }
        }

        impl<'a, 'b, T> Slice<'a, RangeFrom<u32>> for $ty_mut<'b, T> {
            type Output = $ty<'a, T>;

            fn slice(&'a self, r: RangeFrom<u32>) -> $ty<'a, T> {
                self.0.slice(r.start..self.len())
            }
        }

        impl<'a, 'b, T> Slice<'a, RangeFull> for $ty_mut<'b, T> {
            type Output = $ty<'a, T>;

            fn slice(&'a self, _: RangeFull) -> $ty<'a, T> {
                self.0
            }
        }

        impl<'a, 'b, T> Slice<'a, RangeTo<u32>> for $ty_mut<'b, T> {
            type Output = $ty<'a, T>;

            fn slice(&'a self, r: RangeTo<u32>) -> $ty<'a, T> {
                self.0.slice(0..r.end)
            }
        }

        impl<'a, 'b, T> SliceMut<'a, RangeFull> for $ty_mut<'b, T> {
            type Output = $ty_mut<'a, T>;

            fn slice_mut(&'a mut self, _: RangeFull) -> $ty_mut<'a, T> {
                $ty_mut(self.slice(..))
            }
        }

        impl<'a, 'b, T> SliceMut<'a, Range<u32>> for $ty_mut<'b, T> {
            type Output = $ty_mut<'a, T>;

            fn slice_mut(&'a mut self, r: Range<u32>) -> $ty_mut<'a, T> {
                $ty_mut(self.slice(r))
            }
        }

        impl<'a, 'b, T> SliceMut<'a, RangeFrom<u32>> for $ty_mut<'b, T> {
            type Output = $ty_mut<'a, T>;

            fn slice_mut(&'a mut self, r: RangeFrom<u32>) -> $ty_mut<'a, T> {
                $ty_mut(self.slice(r.start..self.len()))
            }
        }

        impl<'a, 'b, T> SliceMut<'a, RangeTo<u32>> for $ty_mut<'b, T> {
            type Output = $ty_mut<'a, T>;

            fn slice_mut(&'a mut self, r: RangeTo<u32>) -> $ty_mut<'a, T> {
                $ty_mut(self.slice(0..r.end))
            }
        }
    };
}

slice!(ColVec, Col, ColMut);
slice!(Diag, DiagMut);
slice!(RowVec, Row, RowMut);

macro_rules! transpose {
    ($($ty:ident $ty_mut:ident $ty_owned:ident; $yt:ident $mut_yt:ident $owned_yt:ident)+) => {
        $(
            impl<T> Transpose for $ty_owned<T> {
                type Output = $owned_yt<T>;

                fn t(self) -> $owned_yt<T> {
                    $owned_yt(self.0)
                }
            }

            impl<'a, T> Transpose for &'a $ty_owned<T> {
                type Output = $yt<'a, T>;

                fn t(self) -> $yt<'a, T> {
                    self.slice(..).t()
                }
            }

            impl<'a, T> Transpose for &'a mut $ty_owned<T> {
                type Output = $mut_yt<'a, T>;

                fn t(self) -> $mut_yt<'a, T> {
                    self.slice_mut(..).t()
                }
            }

            impl<'a, T> Transpose for $ty_mut<'a, T> {
                type Output = $mut_yt<'a, T>;

                fn t(self) -> $mut_yt<'a, T> {
                    $mut_yt($yt((self.0).0))
                }
            }

            impl<'a, 'b, T> Transpose for &'a $ty_mut<'b, T> {
                type Output = $yt<'a, T>;

                fn t(self) -> $yt<'a, T> {
                    self.slice(..).t()
                }
            }

            impl<'a, T> Transpose for $ty<'a, T> {
                type Output = $yt<'a, T>;

                fn t(self) -> $yt<'a, T> {
                    $yt(self.0)
                }
            }
         )+
    }
}

transpose!(Col ColMut ColVec; Row RowMut RowVec);
transpose!(Row RowMut RowVec; Col ColMut ColVec);
