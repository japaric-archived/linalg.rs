use std::cmp;

use common::Stride;
use mat::{Mat,View};
use mat::traits::MatrixShape;
use traits::{Iterable,UnsafeIndex};

// TODO mozilla/rust#13302 Enforce Copy on M
pub struct Diag<M> {
    mat: M,
    diag: int,
}

impl<
    M: Copy + MatrixShape
> Diag<M> {
    #[inline]
    pub fn new(mat: M, diag: int) -> Diag<M> {
        assert!(diag < mat.ncols() as int && diag > -(mat.nrows() as int),
                "diag: out of bounds: {} of {}",
                diag,
                mat.shape())

        Diag { mat: mat, diag: diag }
    }
}

// Collection
impl<
    M: Copy + MatrixShape
> Collection
for Diag<M> {
    #[inline]
    fn len(&self) -> uint {
        if self.diag > 0 {
            cmp::min(self.mat.ncols() - self.diag as uint, self.mat.nrows())
        } else {
            cmp::min(self.mat.nrows() + self.diag as uint, self.mat.ncols())
        }
    }
}

// Index
impl<
    T,
    M: Copy + MatrixShape + UnsafeIndex<(uint, uint), T>
> Index<uint, T>
for Diag<M> {
    #[inline]
    fn index<'a>(&'a self, index: &uint) -> &'a T {
        let size = self.len();

        assert!(*index < size,
                "index: out of bounds: {} of {} ", index, size);

        unsafe { self.unsafe_index(index) }
    }
}

// Iterable
// TODO mozilla/rust#7059 fallback impl
impl<
    'a,
    'b,
    T
> Iterable<'b, T, Stride<'b, T>>
for Diag<&'a Mat<T>> {
    #[inline]
    fn iter(&'b self) -> Stride<'b, T> {
        let size = self.len();
        let start = if self.diag > 0 {
            self.diag as uint
        } else {
            -self.diag as uint * self.mat.ncols()
        };
        let step = self.mat.ncols() + 1;
        let stop = step * (size - 1) + start + 1;

        Stride::new(self.mat.as_slice(),
                    start,
                    stop,
                    step)
    }
}

impl<
    'a,
    'b,
    T
> Iterable<'b, T, Stride<'b, T>>
for Diag<View<&'a Mat<T>>> {
    #[inline]
    fn iter(&'b self) -> Stride<'b, T> {
        let view = self.mat;
        let mat = view.get_ref();
        let (start_row, start_col) = view.start();

        let size = self.len();
        let start = if self.diag > 0 {
            start_row * mat.ncols() + self.diag as uint + start_col
        } else {
            (start_row - self.diag as uint) * mat.ncols() + start_col
        };
        let step = mat.ncols() + 1;
        let stop = step * (size - 1) + start + 1;

        Stride::new(mat.as_slice(),
                    start,
                    stop,
                    step)
    }
}

// UnsafeIndex
impl<
    T,
    M: UnsafeIndex<(uint, uint), T>
> UnsafeIndex<uint, T>
for Diag<M> {
    #[inline]
    unsafe fn unsafe_index<'a>(&'a self, index: &uint) -> &'a T {
        if self.diag > 0 {
            self.mat.unsafe_index(&(*index, *index + self.diag as uint))
        } else {
            self.mat.unsafe_index(&(*index - self.diag as uint, *index))
        }
    }
}
