use std::ops::{Deref, DerefMut, Index, IndexAssign, IndexMut, Range, RangeFrom, RangeTo};
use std::{fmt, mem};

use traits::{Matrix, Transpose};

impl<T> fmt::Debug for ::strided::Col<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Col({:?})", self.deref())
    }
}

impl<T> Deref for ::strided::Col<T> {
    type Target = ::strided::Slice<T>;

    fn deref(&self) -> &::strided::Slice<T> {
        unsafe {
            mem::transmute(self)
        }
    }
}

impl<T> DerefMut for ::strided::Col<T> {
    fn deref_mut(&mut self) -> &mut ::strided::Slice<T> {
        unsafe {
            mem::transmute(self)
        }
    }
}

/// Element indexing: `&col[i]`
impl<T> Index<u32> for ::strided::Col<T> {
    type Output = T;

    fn index(&self, i: u32) -> &T {
        self.deref().index(i)
    }
}

/// Slicing: `&col[a..b]`
impl<T> Index<Range<u32>> for ::strided::Col<T> {
    type Output = ::strided::Col<T>;

    fn index(&self, r: Range<u32>) -> &::strided::Col<T> {
        unsafe {
            mem::transmute(self.deref().index(r))
        }
    }
}

/// Slicing: `&col[a..]`
impl<T> Index<RangeFrom<u32>> for ::strided::Col<T> {
    type Output = ::strided::Col<T>;

    fn index(&self, r: RangeFrom<u32>) -> &::strided::Col<T> {
        self.index(r.start..self.len())
    }
}

/// Slicing: `&col[..b]`
impl<T> Index<RangeTo<u32>> for ::strided::Col<T> {
    type Output = ::strided::Col<T>;

    fn index(&self, r: RangeTo<u32>) -> &::strided::Col<T> {
        self.index(0..r.end)
    }
}

/// Setting an element: `col[i] = x`
impl<T> IndexAssign<u32, T> for ::strided::Col<T> {
    fn index_assign(&mut self, i: u32, rhs: T) {
        *self.index_mut(i) = rhs;
    }
}

/// Mutable element indexing: `&mut col[i]`
impl<T> IndexMut<u32> for ::strided::Col<T> {
    fn index_mut(&mut self, i: u32) -> &mut T {
        self.deref_mut().index_mut(i)
    }
}

/// Mutable slicing: `&mut col[a..b]`
impl<T> IndexMut<Range<u32>> for ::strided::Col<T> {
    fn index_mut(&mut self, r: Range<u32>) -> &mut ::strided::Col<T> {
        unsafe {
            mem::transmute(self.deref_mut().index_mut(r))
        }
    }
}

/// Mutable slicing: `&mut col[a..]`
impl<T> IndexMut<RangeFrom<u32>> for ::strided::Col<T> {
    fn index_mut(&mut self, r: RangeFrom<u32>) -> &mut ::strided::Col<T> {
        let end = self.len();
        self.index_mut(r.start..end)
    }
}

/// Mutable slicing: `&mut col[..b]`
impl<T> IndexMut<RangeTo<u32>> for ::strided::Col<T> {
    fn index_mut(&mut self, r: RangeTo<u32>) -> &mut ::strided::Col<T> {
        self.index_mut(0..r.end)
    }
}

impl<'a, T> IntoIterator for &'a ::strided::Col<T> {
    type Item = &'a T;
    type IntoIter = ::strided::slice::Iter<'a, T>;

    fn into_iter(self) -> ::strided::slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ::strided::Col<T> {
    type Item = &'a mut T;
    type IntoIter = ::strided::slice::IterMut<'a, T>;

    fn into_iter(self) -> ::strided::slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T> Matrix for ::strided::Col<T> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        self.len()
    }

    fn ncols(&self) -> u32 {
        1
    }
}

unsafe impl<T: Send> Send for ::strided::Col<T> {}

unsafe impl<T: Sync> Sync for ::strided::Col<T> {}

impl<'a, T> Transpose for &'a ::strided::Col<T> {
    type Output = &'a ::strided::Row<T>;

    fn t(self) -> &'a ::strided::Row<T> {
        unsafe {
            mem::transmute(self)
        }
    }
}

impl<'a, T> Transpose for &'a mut ::strided::Col<T> {
    type Output = &'a mut ::strided::Row<T>;

    fn t(self) -> &'a mut ::strided::Row<T> {
        unsafe {
            mem::transmute(self)
        }
    }
}
