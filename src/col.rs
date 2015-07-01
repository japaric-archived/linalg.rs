use std::iter::FromIterator;
use std::marker::PhantomData;
use std::num::{One, Zero};
use std::ops::{Deref, DerefMut, Index, IndexAssign, IndexMut, Range, RangeFrom, RangeTo};
use std::raw::FatPtr;
use std::{fat_ptr, fmt, iter, mem, slice};

use cast::From;

use traits::Transpose;
use u31::U31;

impl<T> ::Col<T> {
    /// Creates a column vector from a slice
    pub fn new(slice: &[T]) -> &::Col<T> {
        unsafe {
            mem::transmute(::Vector::new(slice))
        }
    }

    /// Creates a column vector from a mutable slice
    pub fn new_mut(slice: &mut [T]) -> &mut ::Col<T> {
        unsafe {
            mem::transmute(::Vector::new(slice))
        }
    }

    /// Creates a column vector from an owned slice
    pub fn new_owned(elems: Box<[T]>) -> Box<::Col<T>> {
        unsafe {
            let vector = ::Vector::new(&elems);
            mem::forget(elems);
            mem::transmute(vector)
        }
    }

    /// Creates a column vector filled with ones
    pub fn ones(n: u32) -> Box<::Col<T>> where T: Clone + One {
        iter::repeat(T::one()).take(usize::from(n)).collect()
    }

    /// Creates a column vector filled with zeros
    pub fn zeros(n: u32) -> Box<::Col<T>> where T: Clone + Zero {
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
    pub fn repr(&self) -> FatPtr<T, U31> {
        self.0.repr()
    }

    fn as_mat(&self) -> *mut ::Mat<T, ::order::Col> {
        let FatPtr { data, info: len  } = self.repr();

        fat_ptr::new(FatPtr {
            data: data,
            info: ::mat::Info {
                _marker: PhantomData,
                ncols: len,
                nrows: U31::one(),
            }
        })
    }

    unsafe fn deref_raw(&self) -> *mut ::strided::Col<T> {
        mem::transmute(self.0.deref())
    }

    unsafe fn t_raw(&self) -> *mut ::Row<T> {
        mem::transmute(self)
    }
}

impl<'a, T> AsMut<[T]> for ::Col<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<'a, T> AsMut<::Mat<T, ::order::Col>> for ::Col<T> {
    fn as_mut(&mut self) -> &mut ::Mat<T, ::order::Col> {
        unsafe {
            &mut *self.as_mat()
        }
    }
}

impl<'a, T> AsRef<[T]> for ::Col<T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<'a, T> AsRef<::Mat<T, ::order::Col>> for ::Col<T> {
    fn as_ref(&self) -> &::Mat<T, ::order::Col> {
        unsafe {
            &*self.as_mat()
        }
    }
}

impl<T> fmt::Debug for ::Col<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Col({:?})", self.0.as_ref())
    }
}

impl<T> Deref for ::Col<T> {
    type Target = ::strided::Col<T>;

    fn deref(&self) -> &::strided::Col<T> {
        unsafe {
            &*self.deref_raw()
        }
    }
}

impl<T> DerefMut for ::Col<T> {
    fn deref_mut(&mut self) -> &mut ::strided::Col<T> {
        unsafe {
            &mut *self.deref_raw()
        }
    }
}

impl<T> FromIterator<T> for Box<::Col<T>> {
    fn from_iter<I>(it: I) -> Box<::Col<T>> where I: IntoIterator<Item=T> {
        unsafe {
            mem::transmute(Box::<::Vector<T>>::from_iter(it))
        }
    }
}

/// Slicing: `&col[a..b]`
impl<T> Index<Range<u32>> for ::Col<T> {
    type Output = ::Col<T>;

    fn index(&self, r: Range<u32>) -> &::Col<T> {
        unsafe {
            mem::transmute(self.0.slice(r))
        }
    }
}

/// Slicing: `&col[a..]`
impl<T> Index<RangeFrom<u32>> for ::Col<T> {
    type Output = ::Col<T>;

    fn index(&self, r: RangeFrom<u32>) -> &::Col<T> {
        self.index(r.start..self.len())
    }
}

/// Slicing: `&col[..b]`
impl<T> Index<RangeTo<u32>> for ::Col<T> {
    type Output = ::Col<T>;

    fn index(&self, r: RangeTo<u32>) -> &::Col<T> {
        self.index(0..r.end)
    }
}

/// Mutable slicing: `&mut col[a..b]`
impl<T> IndexMut<Range<u32>> for ::Col<T> {
    fn index_mut(&mut self, r: Range<u32>) -> &mut ::Col<T> {
        unsafe {
            mem::transmute(self.0.slice(r))
        }
    }
}

/// Mutable slicing: `&mut col[a..]`
impl<T> IndexMut<RangeFrom<u32>> for ::Col<T> {
    fn index_mut(&mut self, r: RangeFrom<u32>) -> &mut ::Col<T> {
        let end = self.len();
        self.index_mut(r.start..end)
    }
}

/// Mutable slicing: `&mut col[..b]`
impl<T> IndexMut<RangeTo<u32>> for ::Col<T> {
    fn index_mut(&mut self, r: RangeTo<u32>) -> &mut ::Col<T> {
        self.index_mut(0..r.end)
    }
}

// NB All these "forward impls" shouldn't be necessary, but there is a limitation in the index
// overloading code. See rust-lang/rust#26218
/// Element indexing: `&col[i]`
impl<T> Index<u32> for ::Col<T> {
    type Output = T;

    fn index(&self, i: u32) -> &T {
        self.deref().index(i)
    }
}

/// Setting an element: `col[i] = x`
impl<T> IndexAssign<u32, T> for ::Col<T> {
    fn index_assign(&mut self, i: u32, rhs: T) {
        self.deref_mut().index_assign(i, rhs)
    }
}

/// Mutable element indexing: `&mut col[i]`
impl<T> IndexMut<u32> for ::Col<T> {
    fn index_mut(&mut self, i: u32) -> &mut T {
        self.deref_mut().index_mut(i)
    }
}

impl<'a, T> IntoIterator for &'a ::Col<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ::Col<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

unsafe impl<T: Send> Send for ::Col<T> {}

unsafe impl<T: Sync> Sync for ::Col<T> {}

impl<'a, T> Transpose for &'a ::Col<T> {
    type Output = &'a ::Row<T>;

    fn t(self) -> &'a ::Row<T> {
        unsafe {
            &*self.t_raw()
        }
    }
}

impl<'a, T> Transpose for &'a mut ::Col<T> {
    type Output = &'a mut ::Row<T>;

    fn t(self) -> &'a mut ::Row<T> {
        unsafe {
            &mut *self.t_raw()
        }
    }
}

impl<'a, T> Transpose for Box<::Col<T>> {
    type Output = Box<::Row<T>>;

    fn t(self) -> Box<::Row<T>> {
        unsafe {
            let t = self.t_raw();
            mem::forget(self);
            Box::from_raw(t)
        }
    }
}
