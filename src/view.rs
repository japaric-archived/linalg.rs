//! Sub-matrix views

use std::kinds::marker;
use std::{cmp, mem, raw};

use {Col, Diag, Error, MutView, Row, Trans, View, strided};
use traits::{
    Iter, IterMut, Matrix, MatrixCol, MatrixColMut, MatrixCols, MatrixDiag, MatrixDiagMut,
    MatrixMutCols, MatrixMutRows, MatrixRow, MatrixRowMut, MatrixRows, Transpose,
};

/// Immutable sub-matrix iterator
pub struct Items<'a, T> {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    ptr: *const T,
    state: (uint, uint),
    stop: (uint, uint),
    stride: uint,
}

impl<'a, T> Copy for Items<'a, T> {}

/// Mutable sub-matrix iterator
pub struct MutItems<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    ptr: *mut T,
    state: (uint, uint),
    stop: (uint, uint),
    stride: uint,
}

// XXX Is there a faster way to iterate the sub-matrix?
macro_rules! impl_items {
    ($($items:ty -> $item:ty),+,) => {$(
        impl<'a, T> Iterator<$item> for $items {
            fn next(&mut self) -> Option<$item> {
                if self.state.1 == self.stop.1 {
                    None
                } else {
                    let (row, col) = self.state;

                    self.state.0 += 1;

                    if self.state.0 == self.stop.0 {
                        self.state.0 = 0;
                        self.state.1 += 1;
                    }

                    Some(unsafe {
                        mem::transmute(self.ptr.offset((col * self.stride + row) as int))
                    })
                }
            }

            fn size_hint(&self) -> (uint, Option<uint>) {
                let (stop_row, stop_col) = self.stop;
                let (current_row, current_col) = self.state;

                let total = stop_row * stop_col;
                let done = current_row * stop_col + current_col;
                let left = total - done;

                (left, Some(left))
            }
        }
    )+}
}

impl_items! {
    Items<'a, T> -> &'a T,
    MutItems<'a, T> -> &'a mut T,
}

impl<'a, 'b, T> IterMut<'b, &'b mut T, MutItems<'b, T>> for MutView<'a, T> {
    fn iter_mut(&'b mut self) -> MutItems<'b, T> {
        MutItems {
            _contravariant: marker::ContravariantLifetime::<'b>,
            _nosend: marker::NoSend,
            ptr: self.ptr,
            state: (0, 0),
            stop: if self.size.0 == 0 { (0, 0) } else { self.size },
            stride: self.stride,
        }
    }
}

impl<'a, 'b, T> MatrixColMut<'b, &'b mut [T]> for MutView<'a, T> {
    unsafe fn unsafe_col_mut(&mut self, col: uint) -> Col<&'b mut [T]> {
        Col(mem::transmute(raw::Slice {
            data: self.ptr.offset((col * self.stride) as int) as *const T,
            len: self.nrows(),
        }))
    }
}

impl<'a, T> MatrixDiagMut<T> for MutView<'a, T> {
    fn diag_mut<'b>(&'b mut self, diag: int) -> ::Result<Diag<strided::MutSlice<'b, T>>> {
        let (nrows, ncols) = self.size();
        let stride = self.stride;

        if diag > 0 {
            let diag = diag as uint;

            if diag < ncols {
                let ptr = unsafe { self.ptr.offset((diag * stride) as int) } as *const T;
                let len = cmp::min(nrows, ncols - diag);

                Ok(Diag(unsafe { ::Strided::from_parts(ptr, len, stride + 1) }))
            } else {
                Err(Error::NoSuchDiagonal)
            }
        } else {
            let diag = -diag as uint;

            if diag < nrows {
                let ptr = unsafe { self.ptr.offset(diag as int) } as *const T;
                let len = cmp::min(nrows - diag, ncols);

                Ok(Diag(unsafe { ::Strided::from_parts(ptr, len, stride + 1) }))
            } else {
                Err(Error::NoSuchDiagonal)
            }
        }
    }
}

impl<'a, 'b, T> MatrixMutCols<'b> for MutView<'a, T> {}

impl<'a, 'b, T> MatrixMutRows<'b> for MutView<'a, T> {}

impl<'a, 'b, T> MatrixRowMut<'b, strided::MutSlice<'b, T>> for MutView<'a, T> {
    unsafe fn unsafe_row_mut(&mut self, row: uint) -> Row<strided::MutSlice<T>> {
        let ptr = self.ptr.offset(row as int) as *const T;

        Row(::Strided::from_parts(ptr, self.ncols(), self.stride))
    }
}

macro_rules! impls {
    ($($ty:ty),+) => {$(
        impl<'a, T> Matrix for $ty {
            fn size(&self) -> (uint, uint) {
                self.size
            }
        }

        impl<'a, 'b, T> Iter<'b, &'b T, Items<'b, T>> for $ty {
            fn iter(&'b self) -> Items<'b, T> {
                Items {
                    _contravariant: marker::ContravariantLifetime::<'b>,
                    _nosend: marker::NoSend,
                    ptr: self.ptr as *const _,
                    state: (0, 0),
                    stop: if self.size.0 == 0 { (0, 0) } else { self.size },
                    stride: self.stride,
                }
            }
        }

        impl<'a, 'b, T> MatrixCol<'b, &'b [T]> for $ty {
            unsafe fn unsafe_col(&self, col: uint) -> Col<&'b [T]> {
                Col(mem::transmute(raw::Slice {
                    data: self.ptr.offset((col * self.stride) as int) as *const T,
                    len: self.nrows(),
                }))
            }
        }

        impl<'a, 'b, T> MatrixCols<'b> for $ty {}

        impl<'a, T> MatrixDiag<T> for $ty {
            fn diag<'b>(&'b self, diag: int) -> ::Result<Diag<strided::Slice<'b, T>>> {
                let (nrows, ncols) = self.size();
                let stride = self.stride;

                if diag > 0 {
                    let diag = diag as uint;

                    if diag < ncols {
                        let ptr = unsafe { self.ptr.offset((diag * stride) as int) } as *const T;
                        let len = cmp::min(nrows, ncols - diag);

                        Ok(Diag(unsafe { ::Strided::from_parts(ptr, len, stride + 1) }))
                    } else {
                        Err(Error::NoSuchDiagonal)
                    }
                } else {
                    let diag = -diag as uint;

                    if diag < nrows {
                        let ptr = unsafe { self.ptr.offset(diag as int) } as *const T;
                        let len = cmp::min(nrows - diag, ncols);

                        Ok(Diag(unsafe { ::Strided::from_parts(ptr, len, stride + 1) }))
                    } else {
                        Err(Error::NoSuchDiagonal)
                    }
                }
            }
        }

        impl<'a, 'b, T> MatrixRow<'b, strided::Slice<'b, T>> for $ty {
            unsafe fn unsafe_row(&self, row: uint) -> Row<strided::Slice<T>> {
                let ptr = self.ptr.offset(row as int) as *const _;

                Row(::Strided::from_parts(ptr, self.ncols(), self.stride))
            }
        }

        impl<'a, 'b, T> MatrixRows<'b> for $ty {}

        impl<'a, T> Transpose<Trans<$ty>> for $ty {
            fn t(self) -> Trans<$ty> {
                Trans(self)
            }
        })+
    }
}

impls!(MutView<'a, T>, View<'a, T>);
