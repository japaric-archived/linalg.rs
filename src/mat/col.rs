use common::Stride;
use mat::{Mat,View};
use mat::traits::MatrixShape;
// FIXME mozilla/rust#6515 Use std Index
use traits::{Index,Iterable,UnsafeIndex};

// TODO mozilla/rust#13302 Enforce Copy on M
pub struct Col<M> {
    mat: M,
    col: uint,
}

impl<
    M
> Col<M> {
    #[inline]
    pub unsafe fn unsafe_new(mat: M, col: uint) -> Col<M> {
        Col { mat: mat, col: col }
    }
}

impl<
    M: Copy + MatrixShape
> Col<M> {
    #[inline]
    pub fn new(mat: M, col: uint) -> Col<M> {
        assert!(col < mat.ncols(),
                "col: out of bounds: {} of {}",
                col,
                mat.shape())

        Col { mat: mat, col: col }
    }
}

// Container
impl<
    M: Copy + MatrixShape
> Container
for Col<M> {
    #[inline]
    fn len(&self) -> uint {
        self.mat.nrows()
    }
}

// Index
impl<
    T,
    M: Copy + MatrixShape + UnsafeIndex<(uint, uint), T>
> Index<uint, T>
for Col<M> {
    #[inline]
    fn index<'a>(&'a self, row: &uint) -> &'a T {
        let size = self.len();

        assert!(*row < size, "index: out of bounds: {} of {}", row, size);

        unsafe { self.unsafe_index(row) }
    }
}

// Iterable
// TODO mozilla/rust#7059 fallback impl
impl<
    'a,
    'b,
    T
> Iterable<'b, T, Stride<'b, T>>
for Col<&'a Mat<T>> {
    #[inline]
    fn iter(&'b self) -> Stride<'b, T> {
        Stride::new(self.mat.as_slice(),
                    self.col,
                    self.mat.len(),
                    self.mat.ncols())
    }
}

// Iterable
impl<
    'a,
    'b,
    T
> Iterable<'b, T, Stride<'b, T>>
for Col<View<&'a Mat<T>>> {
    #[inline]
    fn iter(&'b self) -> Stride<'b, T> {
        let view = self.mat;
        let mat = view.get_ref();
        let (start_row, start_col) = view.start();

        let start = self.col + start_col + start_row * mat.ncols();
        let step = mat.ncols();
        let stop = step * (self.len() - 1) + start + 1;

        Stride::new(mat.as_slice(), start, stop, step)
    }
}

// UnsafeIndex
impl<
    T,
    M: UnsafeIndex<(uint, uint), T>
> UnsafeIndex<uint, T>
for Col<M> {
    #[inline]
    unsafe fn unsafe_index<'a>(&'a self, row: &uint) -> &'a T {
        self.mat.unsafe_index(&(*row, self.col))
    }
}
