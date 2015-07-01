//! Strided slices

use std::marker::Unsized;
use std::ops::{Index, IndexAssign, IndexMut, Range, RangeFrom, RangeTo};
use std::raw::FatPtr;
use std::{fat_ptr, fmt, mem};

use cast::From;
use extract::Extract;
use u31::U31;

/// Extra information for strided vectors
#[derive(Clone, Copy)]
pub struct Info {
    /// Length of the vector
    pub len: U31,
    /// Stride of the vector
    pub stride: U31,
}

impl<T> ::strided::Vector<T> {
    /// Returns the length of this vector
    pub fn len(&self) -> u32 {
        self.repr().info.len.u32()
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
    pub fn repr(&self) -> FatPtr<T, Info> {
        fat_ptr::repr(self)
    }

    /// Returns the stride of this slice
    pub fn stride(&self) -> u32 {
        self.repr().info.stride.u32()
    }

    fn index_raw(&self, i: u32) -> *mut T {
        let FatPtr { data, info  } = self.repr();

        assert!(i < info.len.u32());

        unsafe {
            data.offset(i as isize * info.stride.isize())
        }
    }

    fn slice_raw(&self, Range { start, end }: Range<u32>) -> *mut ::strided::Vector<T> {
        let FatPtr { data, info } = self.repr();

        assert!(start <= end);
        assert!(end <= info.len.u32());

        fat_ptr::new(FatPtr {
            data: unsafe {
                data.offset(start as isize * info.stride.isize())
            },
            info: Info {
                len: unsafe {
                    U31::from(end - start).extract()
                },
                stride: info.stride,
            }
        })
    }
}

impl<T> fmt::Debug for ::strided::Vector<T> where T: fmt::Debug {
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
impl<T> Index<u32> for ::strided::Vector<T> {
    type Output = T;

    fn index(&self, i: u32) -> &T {
        unsafe {
            &*self.index_raw(i)
        }
    }
}

/// Slicing: `&slice[a..b]`
impl<T> Index<Range<u32>> for ::strided::Vector<T> {
    type Output = ::strided::Vector<T>;

    fn index(&self, r: Range<u32>) -> &::strided::Vector<T> {
        unsafe {
            &*self.slice_raw(r)
        }
    }
}

/// Slicing: `&slice[a..]`
impl<T> Index<RangeFrom<u32>> for ::strided::Vector<T> {
    type Output = ::strided::Vector<T>;

    fn index(&self, r: RangeFrom<u32>) -> &::strided::Vector<T> {
        self.index(r.start..self.len())
    }
}

/// Slicing: `&slice[..b]`
impl<T> Index<RangeTo<u32>> for ::strided::Vector<T> {
    type Output = ::strided::Vector<T>;

    fn index(&self, r: RangeTo<u32>) -> &::strided::Vector<T> {
        self.index(0..r.end)
    }
}

/// Setting an element: `slice[i] = x`
impl<T> IndexAssign<u32, T> for ::strided::Vector<T> {
    fn index_assign(&mut self, i: u32, rhs: T) {
        *self.index_mut(i) = rhs;
    }
}

/// Element indexing: `&mut slice[i]`
impl<T> IndexMut<u32> for ::strided::Vector<T> {
    fn index_mut(&mut self, i: u32) -> &mut T {
        unsafe {
            &mut *self.index_raw(i)
        }
    }
}

/// Mutable slicing: `&mut slice[a..b]`
impl<T> IndexMut<Range<u32>> for ::strided::Vector<T> {
    fn index_mut(&mut self, r: Range<u32>) -> &mut ::strided::Vector<T> {
        unsafe {
            &mut *self.slice_raw(r)
        }
    }
}

/// Mutable slicing: `&mut slice[a..]`
impl<T> IndexMut<RangeFrom<u32>> for ::strided::Vector<T> {
    fn index_mut(&mut self, r: RangeFrom<u32>) -> &mut ::strided::Vector<T> {
        let end = self.len();
        self.index_mut(r.start..end)
    }
}

/// Mutable slicing: `&mut slice[..b]`
impl<T> IndexMut<RangeTo<u32>> for ::strided::Vector<T> {
    fn index_mut(&mut self, r: RangeTo<u32>) -> &mut ::strided::Vector<T> {
        self.index_mut(0..r.end)
    }
}

impl<'a, T> IntoIterator for &'a ::strided::Vector<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ::strided::Vector<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

/// Iterator over a strided slice
pub struct Iter<'a, T: 'a> {
    s: &'a ::strided::Vector<T>,
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        let FatPtr { data, info  } = self.s.repr();

        info.len.checked_sub(1).and_then(|len| unsafe {
            self.s = &*fat_ptr::new(FatPtr {
                data: data,
                info: Info {
                    len: len,
                    stride: info.stride,
                }
            });
            Some(&*data.offset(len.isize() * info.stride.isize()))
        })
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let FatPtr { data, info } = self.s.repr();

        info.len.checked_sub(1).and_then(|len| unsafe {
            self.s = &*fat_ptr::new(FatPtr {
                data: data.offset(info.stride.isize()),
                info: Info {
                    len: len,
                    stride: info.stride,
                }
            });
            Some(&*data)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = usize::from(self.s.len());
        (exact, Some(exact))
    }
}

/// Iterator over a mutable strided slice
pub struct IterMut<'a, T:  'a> {
    s: &'a mut ::strided::Vector<T>,
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<&'a mut T> {
        let FatPtr { data, info  } = self.s.repr();

        info.len.checked_sub(1).and_then(|len| unsafe {
            self.s = &mut *fat_ptr::new(FatPtr {
                data: data,
                info: Info {
                    len: len,
                    stride: info.stride,
                }
            });
            Some(&mut *data.offset(len.isize() * info.stride.isize()))
        })
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        let FatPtr { data, info  } = self.s.repr();

        info.len.checked_sub(1).and_then(|len| unsafe {
            self.s = &mut *fat_ptr::new(FatPtr {
                data: data.offset(info.stride.isize()),
                info: Info {
                    len: len,
                    stride: info.stride,
                }
            });
            Some(&mut *data)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = usize::from(self.s.len());
        (exact, Some(exact))
    }
}

impl<T> Unsized for ::strided::Vector<T> {
    type Data = T;
    type Info = Info;

    fn size_of_val(info: Info) -> usize {
        info.len.usize() * info.stride.usize() * mem::size_of::<T>()
    }
}
