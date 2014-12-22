use std::iter::order;
use std::kinds::marker;
use std::{fmt, mem};

use {Error, Result};
use error::OutOfBounds;

pub struct Items<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    state: *mut T,
    stop: *mut T,
    stride: uint,
}

impl<'a, T> Copy for Items<'a, T> {}

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        if self.state == self.stop {
            None
        } else if mem::size_of::<T>() == 0 {
            self.state = unsafe { mem::transmute(self.state as uint + self.stride) };

            Some(unsafe { mem::transmute(1u) })
        } else {
            let old = self.state;
            self.state = unsafe { self.state.offset(self.stride as int) };

            Some(unsafe { mem::transmute(old) })
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let diff = self.stop as uint - self.state as uint;
        let size = mem::size_of::<T>();
        let exact = diff / (if size == 0 { 1 } else { size } * self.stride);

        (exact, Some(exact))
    }
}

impl<'a, T> DoubleEndedIterator<&'a T> for Items<'a, T> {
    fn next_back(&mut self) -> Option<&'a T> {
        if self.state == self.stop {
            None
        } else if mem::size_of::<T>() == 0 {
            self.stop = unsafe { mem::transmute(self.stop as uint - self.stride) };

            Some(unsafe { mem::transmute(1u) })
        } else {
            self.stop = unsafe { self.stop.offset(-(self.stride as int)) };

            Some(unsafe { mem::transmute(self.stop) })
        }
    }
}

pub struct Slice<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    pub data: *mut T,
    pub len: uint,
    pub stride: uint,
}

impl<'a, T> Slice<'a, T> {
    pub fn  at(&self, index: uint) -> ::std::result::Result<&T, OutOfBounds> {
        if index < self.len {
            Ok(unsafe { mem::transmute(self.data.offset((index * self.stride) as int)) })
        } else {
            Err(OutOfBounds)
        }
    }

    pub fn iter(&self) -> Items<T> {
        Items {
            _contravariant: marker::ContravariantLifetime,
            _nosend: marker::NoSend,
            state: self.data,
            stop: unsafe { self.data.offset((self.len * self.stride) as int) },
            stride: self.stride,
        }
    }

    pub fn len(&self) -> uint {
        self.len
    }

    pub fn slice(&self, start: uint, end: uint) -> Result<Slice<T>> {
        if start > end {
            Err(Error::InvalidSlice)
        } else if end > self.len {
            Err(Error::OutOfBounds)
        } else {
            let stride = self.stride;

            Ok(unsafe { ::From::parts((
                self.data.offset((start * stride) as int) as *const T,
                end - start,
                stride,
            ))})
        }
    }
}

impl<'a, T> Copy for Slice<'a, T> {}

impl<'a, T> ::From<(*const T, uint, uint)> for Slice<'a, T> {
    unsafe fn parts((data, len, stride): (*const T, uint, uint)) -> Slice<'a, T> {
        Slice {
            _contravariant: marker::ContravariantLifetime,
            _nosend: marker::NoSend,
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
            try!(write!(f, "{}", *x))
        }

        try!(write!(f, "]"));

        Ok(())
    }
}
