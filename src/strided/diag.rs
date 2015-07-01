use std::{fmt, mem};
use std::ops::{Deref, DerefMut, Index, IndexAssign, IndexMut, Range, RangeFrom, RangeTo};

impl<T> fmt::Debug for ::strided::Diag<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Diag({:?})", self.deref())
    }
}

impl<T> Deref for ::strided::Diag<T> {
    type Target = ::strided::Vector<T>;

    fn deref(&self) -> &::strided::Vector<T> {
        unsafe {
            mem::transmute(self)
        }
    }
}

impl<T> DerefMut for ::strided::Diag<T> {
    fn deref_mut(&mut self) -> &mut ::strided::Vector<T> {
        unsafe {
            mem::transmute(self)
        }
    }
}

/// Slicing: `&diag[a..b]`
impl<T> Index<Range<u32>> for ::strided::Diag<T> {
    type Output = ::strided::Diag<T>;

    fn index(&self, r: Range<u32>) -> &::strided::Diag<T> {
        unsafe {
            mem::transmute(self.deref().index(r))
        }
    }
}

/// Slicing: `&diag[a..]`
impl<T> Index<RangeFrom<u32>> for ::strided::Diag<T> {
    type Output = ::strided::Diag<T>;

    fn index(&self, r: RangeFrom<u32>) -> &::strided::Diag<T> {
        self.index(r.start..self.len())
    }
}

/// Slicing: `&diag[..b]`
impl<T> Index<RangeTo<u32>> for ::strided::Diag<T> {
    type Output = ::strided::Diag<T>;

    fn index(&self, r: RangeTo<u32>) -> &::strided::Diag<T> {
        self.index(0..r.end)
    }
}

/// Mutable slicing: `&mut diag[a..b]`
impl<T> IndexMut<Range<u32>> for ::strided::Diag<T> {
    fn index_mut(&mut self, r: Range<u32>) -> &mut ::strided::Diag<T> {
        unsafe {
            mem::transmute(self.deref_mut().index_mut(r))
        }
    }
}

/// Mutable slicing: `&mut diag[a..]`
impl<T> IndexMut<RangeFrom<u32>> for ::strided::Diag<T> {
    fn index_mut(&mut self, r: RangeFrom<u32>) -> &mut ::strided::Diag<T> {
        let end = self.len();
        self.index_mut(r.start..end)
    }
}

/// Mutable slicing: `&mut diag[..b]`
impl<T> IndexMut<RangeTo<u32>> for ::strided::Diag<T> {
    fn index_mut(&mut self, r: RangeTo<u32>) -> &mut ::strided::Diag<T> {
        self.index_mut(0..r.end)
    }
}

// NB All these "forward impls" shouldn't be necessary, but there is a limitation in the index
// overloading code. See rust-lang/rust#26218
/// Element indexing: `&diag[i]`
impl<T> Index<u32> for ::strided::Diag<T> {
    type Output = T;

    fn index(&self, i: u32) -> &T {
        self.deref().index(i)
    }
}

/// Setting an element: `diag[i] = x`
impl<T> IndexAssign<u32, T> for ::strided::Diag<T> {
    fn index_assign(&mut self, i: u32, rhs: T) {
        self.deref_mut().index_assign(i, rhs)
    }
}

/// Mutable element indexing: `&mut diag[i]`
impl<T> IndexMut<u32> for ::strided::Diag<T> {
    fn index_mut(&mut self, i: u32) -> &mut T {
        self.deref_mut().index_mut(i)
    }
}

impl<'a, T> IntoIterator for &'a ::strided::Diag<T> {
    type Item = &'a T;
    type IntoIter = ::strided::vector::Iter<'a, T>;

    fn into_iter(self) -> ::strided::vector::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ::strided::Diag<T> {
    type Item = &'a mut T;
    type IntoIter = ::strided::vector::IterMut<'a, T>;

    fn into_iter(self) -> ::strided::vector::IterMut<'a, T> {
        self.iter_mut()
    }
}
