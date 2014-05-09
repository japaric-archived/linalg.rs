pub use self::col::Col;
pub use self::cols::Cols;
pub use self::diag::Diag;
pub use self::row::Row;
pub use self::rows::Rows;

use array::Array;
use array::traits::ArrayShape;
use rand::Rng;
use rand::distributions::IndependentSample;
use self::traits::{MatrixCol,MatrixColIterator,MatrixDiag,MatrixRow,
                   MatrixRowIterator,MatrixShape};
use std::num::{One,Zero,one,zero};
// FIXME mozilla/rust#6515 Use std Index
use traits::{Index,UnsafeIndex};

mod col;
mod cols;
mod diag;
mod row;
mod rows;
pub mod traits;

pub type Mat<T> = Array<(uint, uint), T>;

#[inline]
pub fn from_elem<T: Clone>(shape: (uint, uint), elem: T) -> Mat<T> {
    let (nrows, ncols) = shape;

    unsafe {
        Array::from_raw_parts(Vec::from_elem(nrows * ncols, elem), shape)
    }
}

// TODO fork-join parallelism?
pub fn from_fn<T>(shape: (uint, uint), op: |uint, uint| -> T) -> Mat<T> {
    let (nrows, ncols) = shape;
    let mut v = Vec::with_capacity(nrows * ncols);

    for i in range(0, nrows) {
        for j in range(0, ncols) {
            v.push(op(i, j));
        }
    }

    unsafe { Array::from_raw_parts(v, shape) }
}

#[inline]
pub fn ones<T: Clone + One>(size: (uint, uint)) -> Mat<T> {
    from_elem(size, one())
}

#[inline]
pub fn rand<
    T,
    D: IndependentSample<T>,
    R: Rng
>(shape: (uint, uint), dist: &D, rng: &mut R) -> Mat<T> {
    let (nrows, ncols) = shape;

    unsafe {
        Array::from_raw_parts(
            range(0, nrows * ncols).map(|_| dist.ind_sample(rng)).collect(),
            shape
        )
    }
}

#[inline]
pub fn zeros<T: Clone + Zero>(size: (uint, uint)) -> Mat<T> {
    from_elem(size, zero())
}

// Index
impl<
    T
> Index<(uint, uint), T>
for Mat<T> {
    #[inline]
    fn index<'a>(&'a self, index: &(uint, uint)) -> &'a T {
        let &(row, col) = index;
        let (nrows, ncols) = self.shape();

        assert!(row < nrows && col < ncols,
                "index: out of bounds: {} of {}", index, self.shape());

        unsafe { self.as_slice().unsafe_ref(row * ncols + col) }
    }
}

// MatrixCol
impl<
    'a,
    T
> MatrixCol
for &'a Mat<T> {
    #[inline]
    fn col(self, col: uint) -> Col<&'a Mat<T>> {
        Col::new(self, col)
    }

    #[inline]
    unsafe fn unsafe_col(self, col: uint) -> Col<&'a Mat<T>> {
        Col::unsafe_new(self, col)
    }
}

// MatrixColIterator
impl<
    'a,
    T
> MatrixColIterator
for &'a Mat<T> {
    #[inline]
    fn cols(self) -> Cols<&'a Mat<T>> {
        Cols::new(self)
    }
}

// MatrixDiag
impl<
    'a,
    T
> MatrixDiag
for &'a Mat<T> {
    #[inline]
    fn diag(self, diag: int) -> Diag<&'a Mat<T>> {
        Diag::new(self, diag)
    }
}

// MatrixRow
impl<
    'a,
    T
> MatrixRow
for &'a Mat<T> {
    #[inline]
    fn row(self, row: uint) -> Row<&'a Mat<T>> {
        Row::new(self, row)
    }

    #[inline]
    unsafe fn unsafe_row(self, row: uint) -> Row<&'a Mat<T>> {
        Row::unsafe_new(self, row)
    }
}

// MatrixRowIterator
impl<
    'a,
    T
> MatrixRowIterator
for &'a Mat<T> {
    #[inline]
    fn rows(self) -> Rows<&'a Mat<T>> {
        Rows::new(self)
    }
}

// MatrixShape
impl <
    'a,
    T
> MatrixShape
for &'a Mat<T> {
    #[inline]
    fn ncols(self) -> uint {
        self.shape().val1()
    }

    #[inline]
    fn nrows(self) -> uint {
        self.shape().val0()
    }
}

// UnsafeIndex
impl<
    T
> UnsafeIndex<(uint, uint), T>
for Mat<T> {
    #[inline]
    unsafe fn unsafe_index<'a>(&'a self, index: &(uint, uint)) -> &'a T {
        let &(row, col) = index;
        let (_, ncols) = self.shape();

        self.as_slice().unsafe_ref(row * ncols + col)
    }
}

impl<
    'a,
    T
> UnsafeIndex<(uint, uint), T>
for &'a Mat<T> {
    #[inline]
    unsafe fn unsafe_index<'b>(&'b self, index: &(uint, uint)) -> &'b T {
        (*self).unsafe_index(index)
    }
}
