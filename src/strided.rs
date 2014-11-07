//! Strided slices

use std::fmt::{Formatter, Show, mod};
use std::kinds::marker;
use std::mem;

use blas::copy::BlasCopy;
use blas::{BlasAccelerated, BlasMutPtr, BlasPtr, BlasStride, blasint, to_blasint};
use notsafe::{UnsafeIndex, UnsafeIndexMut, UnsafeMutSlice, UnsafeSlice};
use private::{PrivateIter, PrivateMutIter, PrivateToOwned};
use traits::Collection;

pub use StridedItems as Items;
pub use StridedMutItems as MutItems;
pub use StridedMutSlice as MutSlice;
pub use StridedSlice as Slice;

macro_rules! impl_items {
    ($($items:ty -> $item:ty);+;) => {$(
        impl<'a, T> Iterator<$item> for $items {
            fn next(&mut self) -> Option<$item> {
                if self.state == self.stop {
                    None
                } else if mem::size_of::<T>() == 0 {
                    self.state = unsafe { mem::transmute(self.state as int + self.stride) };

                    Some(unsafe { mem::transmute(1u) })
                } else {
                    let old = self.state;
                    self.state = unsafe { self.state.offset(self.stride) };

                    Some(unsafe { mem::transmute(old) })
                }
            }

            fn size_hint(&self) -> (uint, Option<uint>) {
                let diff = self.stop as uint - self.state as uint;
                let size = mem::size_of::<T>();
                let exact = diff / (if size == 0 { 1 } else { size } * self.stride as uint);

                (exact, Some(exact))
            }
        }

        impl<'a, T> DoubleEndedIterator<$item> for $items {
            fn next_back(&mut self) -> Option<$item> {
                if self.state == self.stop {
                    None
                } else if mem::size_of::<T>() == 0 {
                    self.stop = unsafe { mem::transmute(self.stop as int - self.stride) };

                    Some(unsafe { mem::transmute(1u) })
                } else {
                    self.stop = unsafe { self.stop.offset(-self.stride) };

                    Some(unsafe { mem::transmute(self.stop) })
                }
            }
        }
    )+}
}

impl_items!(
    Items<'a, T> -> &'a T;
    MutItems<'a, T> -> &'a mut T;
)

impl<'a, T> BlasMutPtr<T> for MutSlice<'a, T> where T: BlasAccelerated {
    fn blas_mut_ptr(&mut self) -> *mut T {
        self.data
    }
}

impl<'a, 'b, T> PrivateMutIter<'b, &'b mut T, MutItems<'b, T>> for MutSlice<'a, T> {
    fn private_mut_iter(&'b mut self) -> MutItems<'b, T> {
        MutItems {
            _contravariant: marker::ContravariantLifetime::<'a>,
            _nocopy: marker::NoCopy,
            _nosend: marker::NoSend,
            state: self.data,
            stride: self.stride as int,
            stop: unsafe { self.data.offset((self.len * self.stride) as int) },
        }
    }
}

impl<'a, T> UnsafeIndexMut<uint, T> for MutSlice<'a, T> {
    unsafe fn unsafe_index_mut(&mut self, &index: &uint) -> &mut T {
        mem::transmute(self.data.offset((index * self.stride) as int))
    }
}

impl<'a, 'b, T> UnsafeMutSlice<'b, uint, MutSlice<'b, T>> for MutSlice<'a, T> {
    unsafe fn unsafe_mut_slice(&'b mut self, start: uint, end: uint) -> MutSlice<'b, T> {
        MutSlice {
            _contravariant: marker::ContravariantLifetime::<'b>,
            _nocopy: marker::NoCopy,
            _nosend: marker::NoSend,
            data: self.data.offset(start as int),
            len: end - start,
            stride: self.stride,
        }
    }
}

macro_rules! impl_slice {
    ($($ty:ty),+) => {$(
        impl<'a, T> BlasPtr<T> for $ty where T: BlasAccelerated {
            fn blas_ptr(&self) -> *const T {
                self.data as *const T
            }
        }

        impl<'a, T> BlasStride for $ty where T: BlasAccelerated {
            fn blas_stride(&self) -> blasint {
                to_blasint(self.stride)
            }
        }

        impl<'a, T> Collection for $ty {
            fn len(&self) -> uint {
                self.len
            }
        }

        impl<'a, T> PrivateToOwned<T> for $ty where T: BlasCopy {}

        impl<'a, T> Show for $ty where T: Show {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                try!(write!(f, "["));

                let mut is_first = true;
                for x in self.private_iter() {
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

        impl<'a, T> UnsafeIndex<uint, T> for $ty {
            unsafe fn unsafe_index(&self, &index: &uint) -> &T {
                mem::transmute(self.data.offset((index * self.stride) as int))
            }
        }

        impl<'a, 'b, T> PrivateIter<'b, &'b T, Items<'b, T>> for $ty {
            fn private_iter(&'b self) -> Items<'b, T> {
                Items {
                    _contravariant: marker::ContravariantLifetime::<'b>,
                    _nosend: marker::NoSend,
                    state: self.data as *const T,
                    stride: self.stride as int,
                    stop: unsafe { self.data.offset((self.len * self.stride) as int) as *const T},
                }
            }
        }

        impl<'a, 'b, T> UnsafeSlice<'b, uint, Slice<'b, T>> for $ty {
            unsafe fn unsafe_slice(&'b self, start: uint, end: uint) -> Slice<'b, T> {
                Slice {
                    _contravariant: marker::ContravariantLifetime::<'b>,
                    _nosend: marker::NoSend,
                    data: self.data.offset(start as int) as *const T,
                    len: end - start,
                    stride: self.stride,
                }
            }
        }
    )+}
}

impl_slice!(MutSlice<'a, T>, Slice<'a, T>)
