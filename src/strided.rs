//! Strided slices

use std::kinds::marker;
use std::{fmt, mem};

use Error;
use blas::{MutVector, ToBlasInt, Vector, blasint};
use error::OutOfBounds;
use traits::{At, AtMut, Collection, Iter, IterMut, SliceMut};

/// Iterator over an immutable strided slice
pub struct Items<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    state: *const T,
    stride: uint,
    stop: *const T,
}

/// Iterator over a mutable strided slice
pub struct MutItems<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    state: *mut T,
    stride: uint,
    stop: *mut T,
}

macro_rules! impl_items {
    ($($items:ty -> $item:ty),+,) => {$(
        impl<'a, T> Iterator<$item> for $items {
            fn next(&mut self) -> Option<$item> {
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

        impl<'a, T> DoubleEndedIterator<$item> for $items {
            fn next_back(&mut self) -> Option<$item> {
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
        })+
    }
}

impl_items!{
    MutItems<'a, T> -> &'a mut T,
    Items<'a, T> -> &'a T,
}

/// A mutable strided slice
pub struct MutSlice<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nocopy: marker::NoCopy,
    _nosend: marker::NoSend,
    ptr: *mut T,
    length: uint,
    stride: uint,
}

impl<'a, T> ::Strided<T> for MutSlice<'a, T> {
    unsafe fn from_parts(ptr: *const T, length: uint, stride: uint) -> MutSlice<'a, T> {
        MutSlice {
            _contravariant: marker::ContravariantLifetime,
            _nocopy: marker::NoCopy,
            _nosend: marker::NoSend,
            length: length,
            ptr: ptr as *mut T,
            stride: stride,
        }
    }
}

impl<'a, T> AtMut<uint, T> for MutSlice<'a, T> {
    fn at_mut(&mut self, index: uint) -> Result<&mut T, OutOfBounds> {
        if index < self.length {
            Ok(unsafe { mem::transmute(self.ptr.offset((index * self.stride) as int)) })
        } else {
            Err(OutOfBounds)
        }
    }
}

impl<'a, 'b, T> IterMut<'b, &'b mut T, MutItems<'b, T>> for MutSlice<'a, T> {
    fn iter_mut(&'b mut self) -> MutItems<'b, T> {
        MutItems {
            _contravariant: marker::ContravariantLifetime,
            _nosend: marker::NoSend,
            state: self.ptr,
            stride: self.stride,
            stop: unsafe { self.ptr.offset((self.length * self.stride) as int) },
        }
    }
}

impl<'a, T> MutVector<T> for MutSlice<'a, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<'a, 'b, T> SliceMut<'b, uint, MutSlice<'b, T>> for MutSlice<'a, T> {
    fn slice_mut(&mut self, start: uint, end: uint) -> ::Result<MutSlice<'b, T>> {
        if start > end {
            Err(Error::InvalidSlice)
        } else if end > self.length {
            Err(Error::OutOfBounds)
        } else {
            let stride = self.stride;

            Ok(MutSlice {
                _contravariant: marker::ContravariantLifetime,
                _nocopy: marker::NoCopy,
                _nosend: marker::NoSend,
                length: end - start,
                ptr: unsafe { self.ptr.offset((start * stride) as int) },
                stride: stride,
            })
        }
    }

    fn slice_from_mut(&mut self, start: uint) -> ::Result<MutSlice<T>> {
        let end = self.length;

        SliceMut::slice_mut(self, start, end)
    }

    fn slice_to_mut(&mut self, end: uint) -> ::Result<MutSlice<T>> {
        SliceMut::slice_mut(self, 0, end)
    }
}

/// An immutable strided slice
// XXX I really really wish I could write this as a DST: &Slice<T>, &mut Slice<T>
pub struct Slice<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    length: uint,
    ptr: *const T,
    stride: uint,
}

impl<'a, T> ::Strided<T> for Slice<'a, T> {
    unsafe fn from_parts(ptr: *const T, length: uint, stride: uint) -> Slice<'a, T> {
        Slice {
            _contravariant: marker::ContravariantLifetime,
            _nosend: marker::NoSend,
            length: length,
            ptr: ptr,
            stride: stride,
        }
    }
}

macro_rules! impls {
    ($($ty:ty),+) => {$(
        impl<'a, T> At<uint, T> for $ty {
            fn at(&self, index: uint) -> Result<&T, OutOfBounds> {
                if index < self.length {
                    Ok(unsafe { mem::transmute(self.ptr.offset((index * self.stride) as int)) })
                } else {
                    Err(OutOfBounds)
                }
            }
        }

        impl<'a, 'b, T> Iter<'b, &'b T, Items<'b, T>> for $ty {
            fn iter(&'b self) -> Items<'b, T> {
                Items {
                    _contravariant: marker::ContravariantLifetime,
                    _nosend: marker::NoSend,
                    state: self.ptr as *const _,
                    stride: self.stride,
                    stop: unsafe {
                        self.ptr.offset((self.length * self.stride) as int) as *const _
                    },
                }
            }
        }

        impl<'a, T> Collection for $ty {
            fn len(&self) -> uint {
                self.length
            }
        }

        impl<'a, T> fmt::Show for $ty where T: fmt::Show {
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

        impl<'a, 'b, T> ::traits::Slice<'b, uint, Slice<'b, T>> for $ty {
            fn slice(&self, start: uint, end: uint) -> ::Result<Slice<'b, T>> {
                if start > end {
                    Err(Error::InvalidSlice)
                } else if end > self.length {
                    Err(Error::OutOfBounds)
                } else {
                    let stride = self.stride;

                    Ok(Slice {
                        _contravariant: marker::ContravariantLifetime,
                        _nosend: marker::NoSend,
                        length: end - start,
                        ptr: unsafe { self.ptr.offset((start * stride) as int) } as *const T,
                        stride: stride,
                    })
                }
            }

            fn slice_from(&self, start: uint) -> ::Result<Slice<T>> {
                let end = self.length;

                ::traits::Slice::slice(self, start, end)
            }

            fn slice_to(&self, end: uint) -> ::Result<Slice<T>> {
                ::traits::Slice::slice(self, 0, end)
            }
        }

        impl<'a, T> Vector<T> for $ty {
            fn as_ptr(&self) -> *const T {
                self.ptr as *const _
            }

            fn stride(&self) -> blasint {
                self.stride.to_blasint()
            }
        })+
    }
}

impls!(MutSlice<'a, T>, Slice<'a, T>)
