//! Sub-matrix view

use std::kinds::marker;
use std::mem;
use std::num::{One, Zero, mod};

use blas::{BLAS_NO_TRANS, to_blasint};
use blas::gemm::BlasGemm;
use notsafe::{UnsafeIndex, UnsafeIndexMut};
use traits::{Iter, Matrix, MutIter};
use {Mat, MutView, View};

pub use ViewItems as Items;
pub use ViewMutItems as MutItems;

mod col;
mod cols;
mod diag;
mod row;
mod rows;
mod trans;
mod view;

// XXX Is there a faster way to iterate the sub-matrix?
macro_rules! impl_items {
    ($($items:ty -> $item:ty),+,) => {$(
        impl<'a, T> Iterator<$item> for $items {
            fn next(&mut self) -> Option<$item> {
                if self.state.1 == self.stop.1 {
                    None
                } else {
                    let (row, col) = self.state;

                    *self.state.mut0() += 1;

                    if self.state.0 == self.stop.0 {
                        *self.state.mut0() = 0;
                        *self.state.mut1() += 1;
                    }

                    Some(unsafe {
                        mem::transmute(self.data.offset((col * self.stride + row) as int))
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

impl_items!(
    Items<'a, T> -> &'a T,
    MutItems<'a, T> -> &'a mut T,
)

macro_rules! impl_view {
    ($($ty:ty),+) => {$(
        impl<'a, T> Collection for $ty {
            fn len(&self) -> uint {
                let (nrows, ncols) = self.size;

                nrows * ncols
            }
        }

        impl<'a, 'b, T> Iter<'b, &'b T, Items<'b, T>> for $ty {
            fn iter(&'b self) -> Items<'b, T> {
                Items {
                    _contravariant: marker::ContravariantLifetime::<'b>,
                    _nosend: marker::NoSend,
                    data: self.data as *const T,
                    state: (0, 0),
                    stop: self.size,
                    stride: self.stride,
                }
            }
        }

        impl<'a, T> Matrix for $ty {
            fn size(&self) -> (uint, uint) {
                self.size
            }
        }

        impl<'a, T> UnsafeIndex<(uint, uint), T> for $ty {
            unsafe fn unsafe_index(&self, &(row, col): &(uint, uint)) -> &T {
                mem::transmute(self.data.offset((col * self.stride + row) as int))
            }
        }
    )+}
}

impl_view!(MutView<'a, T>, View<'a, T>)

impl<'a, T> UnsafeIndexMut<(uint, uint), T> for MutView<'a, T> {
    unsafe fn unsafe_index_mut(&mut self, &(row, col): &(uint, uint)) -> &mut T {
        mem::transmute(self.data.offset((col * self.stride + row) as int))
    }
}

impl<'a, 'b, T> MutIter<'b, &'b mut T, MutItems<'b, T>> for MutView<'a, T> {
    fn mut_iter(&'b mut self) -> MutItems<'b, T> {
        MutItems {
            _contravariant: marker::ContravariantLifetime::<'a>,
            _nocopy: marker::NoCopy,
            _nosend: marker::NoSend,
            data: self.data,
            state: (0, 0),
            stop: self.size,
            stride: self.stride,
        }
    }
}

impl<'a, 'b, T> Mul<View<'b, T>, Mat<T>> for View<'a, T> where T: BlasGemm + One + Zero {
    /// - Memory: `O(lhs.nrows * rhs.ncols)`
    /// - Time: `O(lhs.nrows * lhs.ncols * rhs.ncols)`
    ///
    /// # Panics
    ///
    /// Panics if `lhs.ncols != rhs.nrows`
    fn mul(&self, rhs: &View<'b, T>) -> Mat<T> {
        assert!(self.ncols() == rhs.nrows());

        let length = self.nrows().checked_mul(&rhs.ncols()).unwrap();
        let mut data = Vec::with_capacity(length);
        unsafe { data.set_len(length) }

        let gemm = BlasGemm::gemm(None::<T>);
        let transa = &BLAS_NO_TRANS;
        let transb = &BLAS_NO_TRANS;
        let m = &to_blasint(self.nrows());
        let n = &to_blasint(rhs.ncols());
        let k = &to_blasint(self.ncols());
        let alpha = &num::one();
        let a = self.data;
        let lda = &to_blasint(self.stride);
        let b = rhs.data;
        let ldb = &to_blasint(rhs.stride);
        let beta = &num::zero();
        let c = data.as_mut_ptr();
        let ldc = m;
        unsafe { gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }

        Mat::new(data, self.nrows())
    }
}
