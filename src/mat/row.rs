use std::slice::Items;

use mat::{Mat,View};
use mat::traits::MatrixShape;
use traits::{Iterable,UnsafeIndex};

// TODO mozilla/rust#13302 Enforce Copy on M
pub struct Row<M> {
    mat: M,
    row: uint,
}

impl<
    M
> Row<M> {
    #[inline]
    pub unsafe fn unsafe_new(mat: M, row: uint) -> Row<M> {
        Row { mat: mat, row: row }
    }
}

impl<
    M: Copy + MatrixShape
> Row<M> {
    #[inline]
    pub fn new(mat: M, row: uint) -> Row<M> {
        assert!(row < mat.nrows(),
                "row: out of bounds: {} of {}",
                row,
                mat.shape())

        Row { mat: mat, row: row }
    }
}

// Collection
impl<
    M: Copy + MatrixShape
> Collection
for Row<M> {
    #[inline]
    fn len(&self) -> uint {
        self.mat.ncols()
    }
}

// Index
impl<
    T,
    M: Copy + MatrixShape + UnsafeIndex<(uint, uint), T>
> Index<uint, T>
for Row<M> {
    #[inline]
    fn index<'a>(&'a self, col: &uint) -> &'a T {
        let size = self.len();

        assert!(*col < size, "index: out of bounds: {} of {}", col, size);

        unsafe { self.unsafe_index(col) }
    }
}

// Iterable
// TODO mozilla/rust#7059 fallback impl
impl<
    'a,
    'b,
    T
> Iterable<'b, T, Items<'b, T>>
for Row<&'a Mat<T>> {
    #[inline]
    fn iter(&'b self) -> Items<'b, T> {
        self.mat.slice(self.row * self.mat.ncols(),
                       (self.row + 1) * self.mat.ncols()).iter()
    }
}

impl<
    'a,
    'b,
    T
> Iterable<'b, T, Items<'b, T>>
for Row<View<&'a Mat<T>>> {
    #[inline]
    fn iter(&'b self) -> Items<'b, T> {
        let view = self.mat;
        let mat = view.get_ref();
        let (start_row, start_col) = view.start();

        let start = start_col + (start_row + self.row) * mat.ncols();
        let stop = start + self.len();

        mat.slice(start, stop).iter()
    }
}

// UnsafeIndex
impl<
    T,
    M: UnsafeIndex<(uint, uint), T>
> UnsafeIndex<uint, T>
for Row<M> {
    #[inline]
    unsafe fn unsafe_index<'a>(&'a self, col: &uint) -> &'a T {
        self.mat.unsafe_index(&(self.row, *col))
    }
}
