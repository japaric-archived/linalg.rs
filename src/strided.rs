//! Strided slices

use core::nonzero::NonZero;
use std::iter::IntoIterator;
use std::marker::PhantomData;
use std::ops::Range;
use std::{fmt, mem, slice};

use cast::From;
use extract::Extract;

use Slice;
use {ColMut, DiagMut, RowMut};

/// Iterator over an immutable strided slice
// NB `stride` is guaranteed to be positive
pub struct Iter<'a, T> {
    _marker: PhantomData<fn() -> &'a T>,
    state: NonZero<*mut T>,
    stop: NonZero<*mut T>,
    stride: NonZero<i32>,
}

impl<'a, T> Iter<'a, T> {
    unsafe fn new(data: *mut T, len: i32, stride: i32) -> Iter<'a, T> {
        debug_assert!(len >= 0);
        debug_assert!(stride > 0);

        Iter {
            _marker: PhantomData,
            state: NonZero::new(data),
            stop: NonZero::new(data.offset(isize::from(len) * isize::from(stride))),
            stride: NonZero::new(stride),
        }
    }

    unsafe fn raw_next_back(&mut self) -> Option<NonZero<*mut T>> {
        if self.state == self.stop {
            None
        } else if mem::size_of::<T>() == 0 {
            self.stop = NonZero::new({
                (*self.stop as usize - usize::from(*self.stride).extract()) as *mut T
            });

            Some(NonZero::new(1_usize as *mut T))
        } else {
            self.stop = NonZero::new(self.stop.offset(-isize::from(*self.stride)));

            Some(self.stop)
        }
    }

    unsafe fn raw_next(&mut self) -> Option<NonZero<*mut T>> {
        if self.state == self.stop {
            None
        } else if mem::size_of::<T>() == 0 {
            self.state = NonZero::new({
                (*self.state as usize + usize::from(*self.stride).extract()) as *mut T
            });

            Some(NonZero::new(1_usize as *mut T))
        } else {
            let old = self.state;
            self.state = NonZero::new(self.state.offset(isize::from(*self.stride)));

            Some(old)
        }
    }
}

impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Iter<'a, T> {
        Iter {
            ..*self
        }
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        unsafe {
            self.raw_next_back().map(|x| &**x)
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        unsafe {
            self.raw_next().map(|x| &**x)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        unsafe {
            let diff = *self.stop as usize - *self.state as usize;
            let size = mem::size_of::<T>();
            let step = usize::from(*self.stride).extract() * {
                if size == 0 {
                    1
                } else {
                    size
                }
            };
            let exact = diff / step;

            (exact, Some(exact))
        }
    }
}

/// Iterator over a mutable strided slice
pub struct IterMut<'a, T>(Iter<'a, T>);

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<&'a mut T> {
        unsafe {
            self.0.raw_next_back().map(|x| &mut **x)
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        unsafe {
            self.0.raw_next().map(|x| &mut **x)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, T> Slice<'a, T> {
    pub unsafe fn new(data: *mut T, len: i32, stride: i32) -> Slice<'a, T> {
        debug_assert!(len >= 0);
        debug_assert!(stride > 0);

        Slice {
            _marker: PhantomData,
            data: NonZero::new(data),
            len: len,
            stride: NonZero::new(stride),
        }
    }

    pub fn as_slice(&self) -> Option<&[T]> {
        unsafe {
            if *self.stride == 1 {
                Some(slice::from_raw_parts(*self.data, usize::from(self.len).extract()))
            } else {
                None
            }
        }
    }

    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        unsafe {
            if *self.stride == 1 {
                Some(slice::from_raw_parts_mut(*self.data, usize::from(self.len).extract()))
            } else {
                None
            }
        }
    }

    pub fn len(&self) -> u32 {
        unsafe {
            u32::from(self.len).extract()
        }
    }

    pub fn slice(&self, r: Range<u32>) -> Slice<'a, T> {
        unsafe {
            let Range { start, end } = r;
            let len = self.len();

            assert!(start <= end && end <= len);

            let (start, end) = (i32::from(start).extract(), i32::from(end).extract());
            let stride = *self.stride;
            let data = self.data.offset(isize::from(start) * isize::from(stride));

            Slice::new(data, end - start, stride)
        }
    }

    pub unsafe fn raw_index(&self, i: u32) -> *mut T {
        assert!(i < self.len());

        self.data.offset(isize::from(i) * isize::from(*self.stride))
    }
}

impl<'a, T> fmt::Debug for Slice<'a, T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(f.write_str("["));

        let mut is_first = true;

        for x in self.iter() {
            if is_first {
                is_first = false;
            } else {
                try!(f.write_str(", "))
            }

            try!(x.fmt(f))
        }

        f.write_str("]")
    }
}

// NB All `impl`s below this point *shouldn't* be in this module, but intra-crate privacy won't
// let me move these `impl`s anywhere else

impl<'a, T> ColMut<'a, T> {
    /// Returns a "mutable iterator" over the column
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut(self.iter())
    }
}

impl<'a, T> IntoIterator for ColMut<'a, T> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> IterMut<'a, T> {
        IterMut(self.0.into_iter())
    }
}

impl<'a, T> DiagMut<'a, T> {
    /// Returns a "mutable iterator" over the diagonal
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut(self.iter())
    }
}

impl<'a, T> IntoIterator for DiagMut<'a, T> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> IterMut<'a, T> {
        IterMut(self.0.into_iter())
    }
}

impl<'a, T> RowMut<'a, T> {
    /// Returns a "mutable iterator" over the row
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut(self.iter())
    }
}

impl<'a, T> IntoIterator for RowMut<'a, T> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> IterMut<'a, T> {
        IterMut(self.0.into_iter())
    }
}

impl<'a, T> Slice<'a, T> {
    pub fn iter(&self) -> Iter<'a, T> {
        unsafe {
            Iter::new(*self.data, self.len, *self.stride)
        }
    }
}
