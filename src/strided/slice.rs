//! Strided slices

use std::ops::{Index, IndexAssign, IndexMut, Range, RangeFrom, RangeTo};
use std::{fmt, mem};

use cast::From;
use core::nonzero::NonZero;
use extract::Extract;
use u31::U31;

impl<T> ::strided::Slice<T> {
    /// Returns the length of this vector
    pub fn len(&self) -> u32 {
        self.repr().len.u32()
    }

    /// Returns an iterator over the elements of this vector
    pub fn iter(&self) -> Iter<T> {
        Iter {
            s: self,
        }
    }

    /// Returns a mutable iterator over the elements of this vector
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            s: self,
        }
    }

    /// Returns the raw representation of this slice
    pub fn repr(&self) -> ::strided::raw::Slice<T> {
        unsafe {
            mem::transmute(self)
        }
    }

    /// Returns the stride of this slice
    pub fn stride(&self) -> u32 {
        self.repr().stride.u32()
    }

    fn index_raw(&self, i: u32) -> *mut T {
        unsafe {
            let ::strided::raw::Slice { data, len, stride } = self.repr();

            assert!(i < len.u32());


            data.offset(i as isize * stride.isize())
        }
    }

    fn slice_raw(&self, r: Range<u32>) -> *mut ::strided::Slice<T> {
        unsafe {
            let ::strided::raw::Slice { data, len, stride } = self.repr();

            assert!(r.start <= r.end);
            assert!(r.end <= len.u32());

            mem::transmute(::strided::raw::Slice {
                data: NonZero::new(data.offset(r.start as isize * stride.isize())),
                len: U31::from(r.end - r.start).extract(),
                stride: stride,
            })
        }
    }
}

impl<T> fmt::Debug for ::strided::Slice<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(f.write_str("["));

        let mut is_first = true;
        for e in self {
            if is_first {
                is_first = false;
            } else {
                try!(f.write_str(", "));
            }

            try!(write!(f, "{:?}", e));
        }

        f.write_str("]")
    }
}

/// Element indexing: `&slice[i]`
impl<T> Index<u32> for ::strided::Slice<T> {
    type Output = T;

    fn index(&self, i: u32) -> &T {
        unsafe {
            &*self.index_raw(i)
        }
    }
}

/// Slicing: `&slice[a..b]`
impl<T> Index<Range<u32>> for ::strided::Slice<T> {
    type Output = ::strided::Slice<T>;

    fn index(&self, r: Range<u32>) -> &::strided::Slice<T> {
        unsafe {
            &*self.slice_raw(r)
        }
    }
}

/// Slicing: `&slice[a..]`
impl<T> Index<RangeFrom<u32>> for ::strided::Slice<T> {
    type Output = ::strided::Slice<T>;

    fn index(&self, r: RangeFrom<u32>) -> &::strided::Slice<T> {
        self.index(r.start..self.len())
    }
}

/// Slicing: `&slice[..b]`
impl<T> Index<RangeTo<u32>> for ::strided::Slice<T> {
    type Output = ::strided::Slice<T>;

    fn index(&self, r: RangeTo<u32>) -> &::strided::Slice<T> {
        self.index(0..r.end)
    }
}

/// Setting an element: `slice[i] = x`
impl<T> IndexAssign<u32, T> for ::strided::Slice<T> {
    fn index_assign(&mut self, i: u32, rhs: T) {
        *self.index_mut(i) = rhs;
    }
}

/// Element indexing: `&mut slice[i]`
impl<T> IndexMut<u32> for ::strided::Slice<T> {
    fn index_mut(&mut self, i: u32) -> &mut T {
        unsafe {
            &mut *self.index_raw(i)
        }
    }
}

/// Mutable slicing: `&mut slice[a..b]`
impl<T> IndexMut<Range<u32>> for ::strided::Slice<T> {
    fn index_mut(&mut self, r: Range<u32>) -> &mut ::strided::Slice<T> {
        unsafe {
            &mut *self.slice_raw(r)
        }
    }
}

/// Mutable slicing: `&mut slice[a..]`
impl<T> IndexMut<RangeFrom<u32>> for ::strided::Slice<T> {
    fn index_mut(&mut self, r: RangeFrom<u32>) -> &mut ::strided::Slice<T> {
        let end = self.len();
        self.index_mut(r.start..end)
    }
}

/// Mutable slicing: `&mut slice[..b]`
impl<T> IndexMut<RangeTo<u32>> for ::strided::Slice<T> {
    fn index_mut(&mut self, r: RangeTo<u32>) -> &mut ::strided::Slice<T> {
        self.index_mut(0..r.end)
    }
}

impl<'a, T> IntoIterator for &'a ::strided::Slice<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ::strided::Slice<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

/// Iterator over a strided slice
pub struct Iter<'a, T: 'a> {
    s: &'a ::strided::Slice<T>,
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        unsafe {
            let ::strided::raw::Slice { data, len, stride } = self.s.repr();
            len.checked_sub(1).and_then(|len| {
                self.s = mem::transmute(::strided::raw::Slice {
                    data: data,
                    len: len,
                    stride: stride,
                });
                Some(&*data.offset(len.isize() * stride.isize()))
            })
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        unsafe {
            let ::strided::raw::Slice { data, len, stride } = self.s.repr();
            len.checked_sub(1).and_then(|len| {
                self.s = mem::transmute(::strided::raw::Slice {
                    data: NonZero::new(data.offset(stride.isize())),
                    len: len,
                    stride: stride,
                });
                Some(&**data)
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = usize::from(self.s.len());
        (exact, Some(exact))
    }
}

/// Iterator over a mutable strided slice
pub struct IterMut<'a, T:  'a> {
    s: &'a mut ::strided::Slice<T>,
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<&'a mut T> {
        unsafe {
            let ::strided::raw::Slice { data, len, stride } = self.s.repr();
            len.checked_sub(1).and_then(|len| {
                self.s = mem::transmute(::strided::raw::Slice {
                    data: data,
                    len: len,
                    stride: stride,
                });
                Some(&mut *data.offset(len.isize() * stride.isize()))
            })
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        unsafe {
            let ::strided::raw::Slice { data, len, stride } = self.s.repr();
            len.checked_sub(1).and_then(|len| {
                self.s = mem::transmute(::strided::raw::Slice {
                    data: NonZero::new(data.offset(stride.isize())),
                    len: len,
                    stride: stride,
                });
                Some(&mut **data)
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = usize::from(self.s.len());
        (exact, Some(exact))
    }
}
