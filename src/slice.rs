use std::kinds::marker;
use std::mem;
use std::raw::{Repr, mod};

use {Col, Diag, Error, Mat, MutView, Row, View};
use traits::{Slice, SliceMut};

macro_rules! impl_slices {
    ($($ty:ty),+) => {$(
        impl<'a, 'b, T> Slice<'b, uint, &'b [T]> for $ty {
            fn slice(&self, start: uint, end: uint) -> ::Result<&[T]> {
                Slice::slice(*self, start, end)
            }

            fn slice_from(&self, start: uint) -> ::Result<&[T]> {
                Slice::slice(self, start, self.len())
            }

            fn slice_to(&self, end: uint) -> ::Result<&[T]> {
                Slice::slice(self, 0, end)
            }
        })+
    }
}

impl_slices!(&'a [T], &'a mut [T])

macro_rules! from_to {
    () => {
        fn slice_from(&self, start: uint) -> ::Result<&[T]> {
            Slice::slice(self, start, self.len())
        }

        fn slice_to(&self, end: uint) -> ::Result<&[T]> {
            Slice::slice(self, 0, end)
        }
    }
}

macro_rules! from_to_mut {
    () => {
        fn slice_from_mut(&mut self, start: uint) -> ::Result<&mut [T]> {
            let end = self.len();

            SliceMut::slice_mut(self, start, end)
        }

        fn slice_to_mut(&mut self, end: uint) -> ::Result<&mut [T]> {
            SliceMut::slice_mut(self, 0, end)
        }
    }
}

impl<'a, 'b, T> SliceMut<'b, uint, &'b mut [T]> for &'a mut [T] {
    fn slice_mut(&mut self, start: uint, end: uint) -> ::Result<&mut [T]> {
        SliceMut::slice_mut(*self, start, end)
    }

    from_to_mut!()
}

impl<'a, T> Slice<'a, uint, &'a [T]> for Box<[T]> {
    fn slice(&self, start: uint, end: uint) -> ::Result<&[T]> {
        Slice::slice(&**self, start, end)
    }

    from_to!()
}

impl<'a, T> SliceMut<'a, uint, &'a mut [T]> for Box<[T]> {
    fn slice_mut(&mut self, start: uint, end: uint) -> ::Result<&mut [T]> {
        SliceMut::slice_mut(&mut **self, start, end)
    }

    from_to_mut!()
}

impl<'a, T> Slice<'a, uint, &'a [T]> for [T] {
    fn slice(&self, start: uint, end: uint) -> ::Result<&[T]> {
        let raw::Slice { data, len } = self.repr();

        if start > end {
            Err(Error::InvalidSlice)
        } else if end > len {
            Err(Error::OutOfBounds)
        } else {
            Ok(unsafe { mem::transmute(raw::Slice {
                data: data.offset(start as int),
                len: end - start,
            }) })
        }
    }

    from_to!()
}

impl<'a, T> SliceMut<'a, uint, &'a mut [T]> for [T] {
    fn slice_mut(&mut self, start: uint, end: uint) -> ::Result<&mut [T]> {
        let raw::Slice { data, len } = self.repr();

        if start > end {
            Err(Error::InvalidSlice)
        } else if end > len {
            Err(Error::OutOfBounds)
        } else {
            Ok(unsafe { mem::transmute(raw::Slice {
                data: data.offset(start as int),
                len: end - start,
            }) })
        }
    }

    from_to_mut!()
}

macro_rules! impls {
    ($($wrapper:ident $v:ty $s:ty),+,) => {$(
        impl<'a, V, S> Slice<'a, uint, $s> for $v where
            V: Slice<'a, uint, S>,
        {
            fn slice(&'a self, start: uint, end: uint) -> ::Result<$s> {
                self.0.slice(start, end).map(|s| $wrapper(s))
            }

            fn slice_from(&'a self, start: uint) -> ::Result<$s> {
                self.0.slice_from(start).map(|s| $wrapper(s))
            }

            fn slice_to(&'a self, end: uint) -> ::Result<$s> {
                self.0.slice_to(end).map(|s| $wrapper(s))
            }
        }

        impl<'a, V, S> SliceMut<'a, uint, $s> for $v where
            V: SliceMut<'a, uint, S>,
        {
            fn slice_mut(&'a mut self, start: uint, end: uint) -> ::Result<$s> {
                self.0.slice_mut(start, end).map(|s| $wrapper(s))
            }

            fn slice_from_mut(&'a mut self, start: uint) -> ::Result<$s> {
                self.0.slice_from_mut(start).map(|s| $wrapper(s))
            }

            fn slice_to_mut(&'a mut self, end: uint) -> ::Result<$s> {
                self.0.slice_to_mut(end).map(|s| $wrapper(s))
            }
        })+
    }
}

impls!{
    Col Col<V> Col<S>,
    Diag Diag<V> Diag<S>,
    Row Row<V> Row<S>,
}

impl<'a, T> Slice<'a, (uint, uint), View<'a, T>> for Mat<T> {
    fn slice(
        &'a self,
        (start_row, start_col): (uint, uint),
        (end_row, end_col): (uint, uint),
    ) -> ::Result<View<'a, T>> {
        let (nrows, ncols) = self.size;

        if end_col > ncols || end_row > nrows {
            Err(Error::OutOfBounds)
        } else if start_col > end_col || start_row > end_row {
            Err(Error::InvalidSlice)
        } else {
            let stride = nrows;
            let ptr = unsafe {
                self.data.as_ptr().offset((start_col * stride + start_row) as int)
            };

            Ok(View {
                _contravariant: marker::ContravariantLifetime::<'a>,
                _nosend: marker::NoSend,
                ptr: ptr,
                size: (end_row - start_row, end_col - start_col),
                stride: stride,
            })
        }
    }

    fn slice_from(&'a self, start: (uint, uint)) -> ::Result<View<'a, T>> {
        Slice::slice(self, start, self.size)
    }

    fn slice_to(&'a self, end: (uint, uint)) -> ::Result<View<'a, T>> {
        Slice::slice(self, (0, 0), end)
    }
}

impl<'a, T> SliceMut<'a, (uint, uint), MutView<'a, T>> for Mat<T> {
    fn slice_mut(
        &'a mut self,
        (start_row, start_col): (uint, uint),
        (end_row, end_col): (uint, uint),
    ) -> ::Result<MutView<'a, T>> {
        let (nrows, ncols) = self.size;

        if end_col > ncols || end_row > nrows {
            Err(Error::OutOfBounds)
        } else if start_col > end_col || start_row > end_row {
            Err(Error::InvalidSlice)
        } else {
            let stride = nrows;
            let ptr = unsafe {
                self.data.as_mut_ptr().offset((start_col * stride + start_row) as int)
            };

            Ok(MutView {
                _contravariant: marker::ContravariantLifetime::<'a>,
                _nocopy: marker::NoCopy,
                _nosend: marker::NoSend,
                ptr: ptr,
                size: (end_row - start_row, end_col - start_col),
                stride: stride,
            })
        }
    }

    fn slice_from_mut(&'a mut self, start: (uint, uint)) -> ::Result<MutView<'a, T>> {
        let end = self.size;

        SliceMut::slice_mut(self, start, end)
    }

    fn slice_to_mut(&'a mut self, end: (uint, uint)) -> ::Result<MutView<'a, T>> {
        SliceMut::slice_mut(self, (0, 0), end)
    }
}

impl<'a, 'b, T> SliceMut<'b, (uint, uint), MutView<'b, T>> for MutView<'a, T> {
    fn slice_mut(
        &'b mut self,
        (start_row, start_col): (uint, uint),
        (end_row, end_col): (uint, uint),
    ) -> ::Result<MutView<'b, T>> {
        let (nrows, ncols) = self.size;

        if end_col > ncols || end_row > nrows {
            Err(Error::OutOfBounds)
        } else if start_col > end_col || start_row > end_row {
            Err(Error::InvalidSlice)
        } else {
            let stride = self.stride;
            let ptr = unsafe {
                self.ptr.offset((start_col * stride + start_row) as int)
            };

            Ok(MutView {
                _contravariant: marker::ContravariantLifetime::<'a>,
                _nocopy: marker::NoCopy,
                _nosend: marker::NoSend,
                ptr: ptr,
                size: (end_row - start_row, end_col - start_col),
                stride: stride,
            })
        }
    }

    fn slice_from_mut(&'b mut self, start: (uint, uint)) -> ::Result<MutView<'b, T>> {
        let end = self.size;

        SliceMut::slice_mut(self, start, end)
    }

    fn slice_to_mut(&'b mut self, end: (uint, uint)) -> ::Result<MutView<'b, T>> {
        SliceMut::slice_mut(self, (0, 0), end)
    }
}

macro_rules! view {
    ($($ty:ty),+) => {$(
        impl<'a, 'b, T> Slice<'b, (uint, uint), View<'b, T>> for $ty {
            fn slice(
                &'b self,
                (start_row, start_col): (uint, uint),
                (end_row, end_col): (uint, uint),
            ) -> ::Result<View<'b, T>> {
                let (nrows, ncols) = self.size;

                if end_col > ncols || end_row > nrows {
                    Err(Error::OutOfBounds)
                } else if start_col > end_col || start_row > end_row {
                    Err(Error::InvalidSlice)
                } else {
                    let stride = self.stride;
                    let ptr = unsafe {
                        self.ptr.offset((start_col * stride + start_row) as int) as *const T
                    };

                    Ok(View {
                        _contravariant: marker::ContravariantLifetime::<'a>,
                        _nosend: marker::NoSend,
                        ptr: ptr,
                        size: (end_row - start_row, end_col - start_col),
                        stride: stride,
                    })
                }
            }

            fn slice_from(&'b self, start: (uint, uint)) -> ::Result<View<'b, T>> {
                Slice::slice(self, start, self.size)
            }

            fn slice_to(&'b self, end: (uint, uint)) -> ::Result<View<'b, T>> {
                Slice::slice(self, (0, 0), end)
            }
        })+
    }
}

view!(MutView<'a, T>, View<'a, T>)
