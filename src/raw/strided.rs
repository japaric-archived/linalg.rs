use std::iter::order;
use std::marker;
use std::{fmt, mem};

use {Error, Result};
use error::OutOfBounds;

pub struct Items<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    state: *mut T,
    stop: *mut T,
    stride: usize,
}

impl<'a, T> Copy for Items<'a, T> {}

impl<'a, T> Iterator for Items<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.state == self.stop {
            None
        } else if mem::size_of::<T>() == 0 {
            self.state = unsafe { mem::transmute(self.state as usize + self.stride) };

            Some(unsafe { mem::transmute(1us) })
        } else {
            let old = self.state;
            self.state = unsafe { self.state.offset(self.stride as isize) };

            Some(unsafe { mem::transmute(old) })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let diff = self.stop as usize - self.state as usize;
        let size = mem::size_of::<T>();
        let exact = diff / (if size == 0 { 1 } else { size } * self.stride);

        (exact, Some(exact))
    }
}

impl<'a, T> DoubleEndedIterator for Items<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        if self.state == self.stop {
            None
        } else if mem::size_of::<T>() == 0 {
            self.stop = unsafe { mem::transmute(self.stop as usize - self.stride) };

            Some(unsafe { mem::transmute(1us) })
        } else {
            self.stop = unsafe { self.stop.offset(-(self.stride as isize)) };

            Some(unsafe { mem::transmute(self.stop) })
        }
    }
}

pub struct Slice<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    pub data: *mut T,
    pub len: usize,
    pub stride: usize,
}

impl<'a, T> Slice<'a, T> {
    pub fn  at(&self, index: usize) -> ::std::result::Result<&T, OutOfBounds> {
        if index < self.len {
            Ok(unsafe { mem::transmute(self.data.offset((index * self.stride) as isize)) })
        } else {
            Err(OutOfBounds)
        }
    }

    pub fn iter(&self) -> Items<T> {
        Items {
            _contravariant: marker::ContravariantLifetime,
            state: self.data,
            stop: unsafe { self.data.offset((self.len * self.stride) as isize) },
            stride: self.stride,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn slice(&self, start: usize, end: usize) -> Result<Slice<T>> {
        if start > end {
            Err(Error::InvalidSlice)
        } else if end > self.len {
            Err(Error::OutOfBounds)
        } else {
            let stride = self.stride;

            Ok(unsafe { ::From::parts((
                self.data.offset((start * stride) as isize) as *const T,
                end - start,
                stride,
            ))})
        }
    }
}

impl<'a, T> Copy for Slice<'a, T> {}

impl<'a, T> ::From<(*const T, usize, usize)> for Slice<'a, T> {
    unsafe fn parts((data, len, stride): (*const T, usize, usize)) -> Slice<'a, T> {
        Slice {
            _contravariant: marker::ContravariantLifetime,
            data: data as *mut _,
            len: len,
            stride: stride,
        }
    }
}

impl<'a, 'b, T, U> PartialEq<Slice<'a, T>> for Slice<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Slice<'a, T>) -> bool {
        self.len == rhs.len && order::eq(self.iter(), rhs.iter())
    }
}

impl<'a, T> fmt::Show for Slice<'a, T> where T: fmt::Show {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "["));

        let mut is_first = true;
        for x in self.iter() {
            if is_first {
                is_first = false;
            } else {
                try!(write!(f, ", "));
            }
            try!(write!(f, "{:?}", *x))
        }

        try!(write!(f, "]"));

        Ok(())
    }
}
