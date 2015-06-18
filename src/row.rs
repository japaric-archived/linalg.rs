use std::iter::FromIterator;
use std::marker::PhantomData;
use std::num::{One, Zero};
use std::ops::{Deref, DerefMut, Index, IndexAssign, IndexMut, Range, RangeFrom, RangeTo};
use std::{fmt, iter, mem, slice};

use cast::From;
use core::nonzero::NonZero;

use traits::{Matrix, Transpose};
use u31::U31;

impl<T> ::Row<T> {
    /// Creates a row vector from a slice
    pub fn new(slice: &[T]) -> &::Row<T> {
        unsafe {
            mem::transmute(::raw::Slice::from(slice))
        }
    }

    /// Creates a row vector from a mutable slice
    pub fn new_mut(slice: &mut [T]) -> &mut ::Row<T> {
        unsafe {
            mem::transmute(::raw::Slice::from(slice))
        }
    }

    /// Creates a row vector from an owned slice
    pub fn new_owned(mut elems: Box<[T]>) -> Box<::Row<T>> {
        unsafe {
            let len = U31::from(elems.len()).unwrap();
            let data = NonZero::new(elems.as_mut_ptr());

            mem::forget(elems);

            mem::transmute(::raw::Slice { data: data, len: len })
        }
    }

    /// Creates a row vector filled with ones
    pub fn ones(n: u32) -> Box<::Row<T>> where T: Clone + One {
        iter::repeat(T::one()).take(usize::from(n)).collect()
    }

    /// Creates a row vector filled with zeros
    pub fn zeros(n: u32) -> Box<::Row<T>> where T: Clone + Zero {
        iter::repeat(T::zero()).take(usize::from(n)).collect()
    }

    /// Returns an iterator over the elements of the vector
    pub fn iter(&self) -> slice::Iter<T> {
        let slice: &[T] = self.as_ref();
        slice.iter()
    }

    /// Returns a mutable iterator over the elements of the vector
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        let slice: &mut [T] = self.as_mut();
        slice.iter_mut()
    }

    /// Returns the raw representation of this vector
    pub fn repr(&self) -> ::raw::Slice<T> {
        unsafe {
            mem::transmute(self)
        }
    }

    fn as_mat_raw(&self) -> *mut ::Mat<T, ::order::Row> {
        unsafe {
            let ::raw::Slice { data, len } = self.repr();
            mem::transmute(::raw::Mat {
                data: data,
                marker: PhantomData::<::order::Row>,
                ncols: len,
                nrows: U31::one(),
            })
        }
    }

    fn deref_raw(&self) -> *mut ::strided::Row<T> {
        unsafe {
            let ::raw::Slice { data, len } = self.repr();

            mem::transmute(::strided::raw::Slice { data: data, len: len, stride: U31::one() })
        }
    }

    fn t_raw(&self) -> *mut ::Col<T> {
        unsafe {
            mem::transmute(self.repr())
        }
    }
}

impl<'a, T> AsMut<[T]> for ::Row<T> {
    fn as_mut(&mut self) -> &mut [T] {
        unsafe {
            &mut *self.repr().as_slice_raw()
        }
    }
}

impl<'a, T> AsMut<::Mat<T, ::order::Row>> for ::Row<T> {
    fn as_mut(&mut self) -> &mut ::Mat<T, ::order::Row> {
        unsafe {
            &mut *self.as_mat_raw()
        }
    }
}

impl<'a, T> AsRef<[T]> for ::Row<T> {
    fn as_ref(&self) -> &[T] {
        unsafe {
            &*self.repr().as_slice_raw()
        }
    }
}

impl<'a, T> AsRef<::Mat<T, ::order::Row>> for ::Row<T> {
    fn as_ref(&self) -> &::Mat<T, ::order::Row> {
        unsafe {
            &*self.as_mat_raw()
        }
    }
}

impl<T> fmt::Debug for ::Row<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            write!(f, "Row({:?})", &*self.repr().as_slice_raw())
        }
    }
}

impl<T> Deref for ::Row<T> {
    type Target = ::strided::Row<T>;

    fn deref(&self) -> &::strided::Row<T> {
        unsafe {
            &*self.deref_raw()
        }
    }
}

impl<T> DerefMut for ::Row<T> {
    fn deref_mut(&mut self) -> &mut ::strided::Row<T> {
        unsafe {
            &mut *self.deref_raw()
        }
    }
}

impl<T> Drop for ::Row<T> {
    fn drop(&mut self) {
        unsafe {
            let ::raw::Slice { data, len, .. } = self.repr();

            if !data.is_null() && *data as usize != mem::POST_DROP_USIZE {
                let len = len.usize();

                mem::drop(Vec::from_raw_parts(*data, len, len))
            }
        }
    }
}

impl<T> FromIterator<T> for Box<::Row<T>> {
    fn from_iter<I>(it: I) -> Box<::Row<T>> where I: IntoIterator<Item=T> {
        unsafe {
            let mut v: Vec<_> = it.into_iter().collect();
            let len = U31::from(v.len()).unwrap();
            let data = NonZero::new(v.as_mut_ptr());

            mem::forget(v);

            mem::transmute(::raw::Slice { data: data, len: len })
        }
    }
}

/// Slicing: `&row[a..b]`
impl<T> Index<Range<u32>> for ::Row<T> {
    type Output = ::Row<T>;

    fn index(&self, r: Range<u32>) -> &::Row<T> {
        unsafe {
            mem::transmute(self.repr().slice(r))
        }
    }
}

/// Slicing: `&row[a..]`
impl<T> Index<RangeFrom<u32>> for ::Row<T> {
    type Output = ::Row<T>;

    fn index(&self, r: RangeFrom<u32>) -> &::Row<T> {
        self.index(r.start..self.len())
    }
}

/// Slicing: `&row[..b]`
impl<T> Index<RangeTo<u32>> for ::Row<T> {
    type Output = ::Row<T>;

    fn index(&self, r: RangeTo<u32>) -> &::Row<T> {
        self.index(0..r.end)
    }
}

/// Mutable slicing: `&mut row[a..b]`
impl<T> IndexMut<Range<u32>> for ::Row<T> {
    fn index_mut(&mut self, r: Range<u32>) -> &mut ::Row<T> {
        unsafe {
            mem::transmute(self.repr().slice(r))
        }
    }
}

/// Mutable slicing: `&mut row[a..]`
impl<T> IndexMut<RangeFrom<u32>> for ::Row<T> {
    fn index_mut(&mut self, r: RangeFrom<u32>) -> &mut ::Row<T> {
        let end = self.len();
        self.index_mut(r.start..end)
    }
}

/// Mutable slicing: `&mut row[..b]`
impl<T> IndexMut<RangeTo<u32>> for ::Row<T> {
    fn index_mut(&mut self, r: RangeTo<u32>) -> &mut ::Row<T> {
        self.index_mut(0..r.end)
    }
}

// NB All these "forward impls" shouldn't be necessary, but there is a limitation in the index
// overloading code. See rust-lang/rust#26218
/// Element indexing: `&row[i]`
impl<T> Index<u32> for ::Row<T> {
    type Output = T;

    fn index(&self, i: u32) -> &T {
        self.deref().index(i)
    }
}

/// Setting an element: `row[i] = x`
impl<T> IndexAssign<u32, T> for ::Row<T> {
    fn index_assign(&mut self, i: u32, rhs: T) {
        self.deref_mut().index_assign(i, rhs)
    }
}

/// Mutable element indexing: `&mut row[i]`
impl<T> IndexMut<u32> for ::Row<T> {
    fn index_mut(&mut self, i: u32) -> &mut T {
        self.deref_mut().index_mut(i)
    }
}

impl<'a, T> IntoIterator for &'a ::Row<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ::Row<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T> Matrix for ::Row<T> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        1
    }

    fn ncols(&self) -> u32 {
        self.len()
    }
}

unsafe impl<T: Send> Send for ::Row<T> {}

unsafe impl<T: Sync> Sync for ::Row<T> {}

impl<'a, T> Transpose for &'a ::Row<T> {
    type Output = &'a ::Col<T>;

    fn t(self) -> &'a ::Col<T> {
        unsafe {
            &*self.t_raw()
        }
    }
}

impl<'a, T> Transpose for &'a mut ::Row<T> {
    type Output = &'a mut ::Col<T>;

    fn t(self) -> &'a mut ::Col<T> {
        unsafe {
            &mut *self.t_raw()
        }
    }
}

impl<'a, T> Transpose for Box<::Row<T>> {
    type Output = Box<::Col<T>>;

    fn t(self) -> Box<::Col<T>> {
        unsafe {
            let t = self.t_raw();
            mem::forget(self);
            mem::transmute(t)
        }
    }
}
