use mat::Mat;
use mat::traits::MatrixShape;
use std::slice::Items;
// FIXME mozilla/rust#6515 Use std Index
use traits::{Index,Iterable,UnsafeIndex};

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

// Index
impl<
    T,
    M: Copy + MatrixShape + UnsafeIndex<(uint, uint), T>
> Index<uint, T>
for Row<M> {
    #[inline]
    fn index<'a>(&'a self, col: &uint) -> &'a T {
        assert!(*col < self.mat.ncols(),
                "index: out of bounds: {} of {}", col, self.mat.shape());

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
