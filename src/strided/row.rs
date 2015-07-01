use std::ops::{Deref, DerefMut, Index, IndexAssign, IndexMut, Range, RangeFrom, RangeTo};
use std::{fmt, mem};

use traits::{Matrix, Transpose};

impl<T> fmt::Debug for ::strided::Row<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Row({:?})", self.deref())
    }
}

impl<T> Deref for ::strided::Row<T> {
    type Target = ::strided::Vector<T>;

    fn deref(&self) -> &::strided::Vector<T> {
        unsafe {
            mem::transmute(self)
        }
    }
}

impl<T> DerefMut for ::strided::Row<T> {
    fn deref_mut(&mut self) -> &mut ::strided::Vector<T> {
        unsafe {
            mem::transmute(self)
        }
    }
}

/// Slicing: `&row[a..b]`
impl<T> Index<Range<u32>> for ::strided::Row<T> {
    type Output = ::strided::Row<T>;

    fn index(&self, r: Range<u32>) -> &::strided::Row<T> {
        unsafe {
            mem::transmute(self.deref().index(r))
        }
    }
}

/// Slicing: `&row[a..]`
impl<T> Index<RangeFrom<u32>> for ::strided::Row<T> {
    type Output = ::strided::Row<T>;

    fn index(&self, r: RangeFrom<u32>) -> &::strided::Row<T> {
        self.index(r.start..self.len())
    }
}

/// Slicing: `&row[..b]`
impl<T> Index<RangeTo<u32>> for ::strided::Row<T> {
    type Output = ::strided::Row<T>;

    fn index(&self, r: RangeTo<u32>) -> &::strided::Row<T> {
        self.index(0..r.end)
    }
}

/// Mutable slicing: `&mut row[a..b]`
impl<T> IndexMut<Range<u32>> for ::strided::Row<T> {
    fn index_mut(&mut self, r: Range<u32>) -> &mut ::strided::Row<T> {
        unsafe {
            mem::transmute(self.deref_mut().index_mut(r))
        }
    }
}

/// Mutable slicing: `&mut row[a..]`
impl<T> IndexMut<RangeFrom<u32>> for ::strided::Row<T> {
    fn index_mut(&mut self, r: RangeFrom<u32>) -> &mut ::strided::Row<T> {
        let end = self.len();
        self.index_mut(r.start..end)
    }
}

/// Mutable slicing: `&mut row[..b]`
impl<T> IndexMut<RangeTo<u32>> for ::strided::Row<T> {
    fn index_mut(&mut self, r: RangeTo<u32>) -> &mut ::strided::Row<T> {
        self.index_mut(0..r.end)
    }
}

// NB All these "forward impls" shouldn't be necessary, but there is a limitation in the index
// overloading code. See rust-lang/rust#26218
/// Element indexing: `&row[i]`
impl<T> Index<u32> for ::strided::Row<T> {
    type Output = T;

    fn index(&self, i: u32) -> &T {
        self.deref().index(i)
    }
}

/// Setting an element: `row[i] = x`
impl<T> IndexAssign<u32, T> for ::strided::Row<T> {
    fn index_assign(&mut self, i: u32, rhs: T) {
        *self.index_mut(i) = rhs;
    }
}

/// Mutable element indexing: `&mut row[i]`
impl<T> IndexMut<u32> for ::strided::Row<T> {
    fn index_mut(&mut self, i: u32) -> &mut T {
        self.deref_mut().index_mut(i)
    }
}

impl<'a, T> IntoIterator for &'a ::strided::Row<T> {
    type Item = &'a T;
    type IntoIter = ::strided::vector::Iter<'a, T>;

    fn into_iter(self) -> ::strided::vector::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ::strided::Row<T> {
    type Item = &'a mut T;
    type IntoIter = ::strided::vector::IterMut<'a, T>;

    fn into_iter(self) -> ::strided::vector::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T> Matrix for ::strided::Row<T> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        1
    }

    fn ncols(&self) -> u32 {
        self.len()
    }
}

unsafe impl<T: Send> Send for ::strided::Row<T> {}

unsafe impl<T: Sync> Sync for ::strided::Row<T> {}

impl<'a, T> Transpose for &'a ::strided::Row<T> {
    type Output = &'a ::strided::Col<T>;

    fn t(self) -> &'a ::strided::Col<T> {
        unsafe {
            mem::transmute(self)
        }
    }
}

impl<'a, T> Transpose for &'a mut ::strided::Row<T> {
    type Output = &'a mut ::strided::Col<T>;

    fn t(self) -> &'a mut ::strided::Col<T> {
        unsafe {
            mem::transmute(self)
        }
    }
}
