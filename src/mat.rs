use std::iter::IntoIterator;
use std::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo};
use std::ptr::Unique;
use std::{iter, mem, slice};

use cast::From;
use extract::Extract;
use onezero::{One, Zero};

use traits::{
    HSplit, HSplitMut, Iter, IterMut, Matrix, MatrixCol, MatrixColMut, MatrixCols, MatrixColsMut,
    MatrixDiag, MatrixDiagMut, MatrixHStripes, MatrixHStripesMut, MatrixRow, MatrixRowMut,
    MatrixRows, MatrixRowsMut, MatrixVStripes, MatrixVStripesMut, Slice, SliceMut, Transpose,
    VSplit, VSplitMut,
};
use {
    Col, ColMut, Cols, Diag, HStripes, Mat, Row, RowMut, Rows, Transposed, VStripes, SubMat,
    SubMatMut,
};

impl<T> Mat<T> {
    /// Creates an owned matrix from its "raw parts"
    ///
    /// NOTE The matrix stores its elements in column major order.
    ///
    /// For example, to produce the following matrix:
    ///
    /// ``` text
    /// [0 1 2 3]
    /// [4 5 6 7]
    /// ```
    ///
    /// This functions must be called with the following inputs:
    ///
    /// ``` text
    /// elems = [0, 4, 1, 5, 2, 6, 3, 7]
    /// nrows = 2
    /// ncols = 4
    /// ```
    ///
    /// # Safety
    ///
    /// User must ensure that:
    ///
    /// - `nrows > 0 &&`
    /// - `ncols > 0 &&`
    /// - `elems.len() == nrows * ncols`
    pub unsafe fn from_raw_parts(mut elems: Box<[T]>, (nrows, ncols): (i32, i32)) -> Mat<T> {
        debug_assert!(nrows >= 0);
        debug_assert!(ncols >= 0);
        debug_assert!(Some(elems.len()) == {
            usize::from(nrows).extract().checked_mul(usize::from(ncols).extract())
        });

        let data = elems.as_mut_ptr();
        mem::forget(elems);

        Mat {
            data: Unique::new(data),
            ncols: ncols,
            nrows: nrows,
        }
    }

    /// Creates an owned matrix with dimensions `(nrows, ncols)` filled with `elem` values
    ///
    /// # Panics
    ///
    /// If:
    ///
    /// - `nrows > 2^31 ||`
    /// - `ncols > 2^31 ||`
    /// - `nrows * ncols > usize::max_value()`
    pub fn from_elem((nrows, ncols): (u32, u32), elem: T) -> Mat<T> where T: Clone {
        unsafe {
            let n = usize::from(nrows).checked_mul(usize::from(ncols)).unwrap();
            let (nrows, ncols) = (i32::from(nrows).unwrap(), i32::from(ncols).unwrap());

            let elems = iter::repeat(elem).take(n).collect::<Vec<_>>().into_boxed_slice();

            Mat::from_raw_parts(elems, (nrows, ncols))
        }
    }

    /// Creates an owned matrix with dimensions `(nrows, ncols)` where each element gets
    /// initialized using the function `f`
    ///
    /// # Panics
    ///
    /// If:
    ///
    /// - `nrows > 2^31 ||`
    /// - `ncols > 2^31 ||`
    /// - `nrows * ncols > usize::max_value()`
    pub fn from_fn<F>((nrows, ncols): (u32, u32), mut f: F) -> Mat<T> where
        F: FnMut((u32, u32)) -> T,
    {
        unsafe {
            let n = usize::from(nrows).checked_mul(usize::from(ncols)).unwrap();
            let (nrows_, ncols_) = (i32::from(nrows).unwrap(), i32::from(ncols).unwrap());

            let mut v = Vec::with_capacity(n);

            for col in 0..ncols {
                for row in 0..nrows {
                    v.push(f((row, col)));
                }
            }

            Mat::from_raw_parts(v.into_boxed_slice(), (nrows_, ncols_))
        }
    }

    /// Creates an owned matrix with dimensions `(nrows, ncols)` filled with ones
    ///
    /// # Panics
    ///
    /// If:
    ///
    /// - `nrows > 2^31 ||`
    /// - `ncols > 2^31 ||`
    /// - `nrows * ncols > usize::max_value()`
    pub fn ones((nrows, ncols): (u32, u32)) -> Mat<T> where T: Clone + One {
        Mat::from_elem((nrows, ncols), T::one())
    }

    /// Creates an owned matrix with dimensions `(nrows, ncols)` filled with zeros
    ///
    /// # Panics
    ///
    /// If:
    ///
    /// - `nrows > 2^31 ||`
    /// - `ncols > 2^31 ||`
    /// - `nrows * ncols > usize::max_value()`
    pub fn zeros((nrows, ncols): (u32, u32)) -> Mat<T> where T: Clone + Zero {
        Mat::from_elem((nrows, ncols), T::zero())
    }

    fn len(&self) -> usize {
        unsafe {
            usize::from(self.nrows).extract() * usize::from(self.ncols).extract()
        }
    }

    unsafe fn raw_index(&self, (row, col): (u32, u32)) -> *mut T {
        assert!(row < self.nrows() && col < self.ncols());

        self.data.offset(isize::from(col) * isize::from(self.nrows()) + isize::from(row))
    }
}

impl<T> HSplit for Mat<T> {
    fn hsplit_at(&self, i: u32) -> (SubMat<T>, SubMat<T>) {
        unsafe {
            let v: *const SubMat<T> = &self.slice(..);
            (*v).hsplit_at(i)
        }
    }
}

impl<T> HSplitMut for Mat<T> {
    fn hsplit_at_mut(&mut self, i: u32) -> (SubMatMut<T>, SubMatMut<T>) {
        unsafe {
            let v: *mut SubMatMut<T> = &mut self.slice_mut(..);
            (*v).hsplit_at_mut(i)
        }
    }
}

impl<'a, T> Index<(u32, u32)> for Mat<T> {
    type Output = T;

    fn index(&self, (row, col): (u32, u32)) -> &T {
        unsafe {
            &*self.raw_index((row, col))
        }
    }
}

impl<'a, T> IndexMut<(u32, u32)> for Mat<T> {
    fn index_mut(&mut self, (row, col): (u32, u32)) -> &mut T {
        unsafe {
            &mut *self.raw_index((row, col))
        }
    }
}

impl<T> Clone for Mat<T> where T: Clone {
    fn clone(&self) -> Mat<T> {
        unsafe {
            let mut v = slice::from_raw_parts(*self.data, self.len()).to_vec();
            let data = v.as_mut_ptr();
            mem::forget(v);

            Mat {
                data: Unique::new(data),
                ..*self
            }
        }
    }
}

impl<T> Drop for Mat<T> {
    fn drop(&mut self) {
        unsafe {
            let ptr = *self.data;

            if !ptr.is_null() && ptr as usize != mem::POST_DROP_USIZE {
                let len = self.len();

                mem::drop(Vec::from_raw_parts(ptr, len, len))
            }
        }
    }
}

impl<'a, T> IntoIterator for &'a Mat<T> {
    type IntoIter = slice::Iter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Mat<T> {
    type IntoIter = slice::IterMut<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<'a, T> Iter<'a> for Mat<T> {
    type Iter = slice::Iter<'a, T>;

    fn iter(&'a self) -> slice::Iter<'a, T> {
        unsafe {
            slice::from_raw_parts(*self.data, self.len()).iter()
        }
    }
}

impl<'a, T> IterMut<'a> for Mat<T> {
    type IterMut  = slice::IterMut<'a, T>;

    fn iter_mut(&'a mut self) -> slice::IterMut<'a, T> {
        unsafe {
            slice::from_raw_parts_mut(*self.data, self.len()).iter_mut()
        }
    }
}

impl<T> Matrix for Mat<T> {
    type Elem = T;

    fn ncols(&self) -> u32 {
        unsafe {
            u32::from(self.ncols).extract()
        }
    }

    fn nrows(&self) -> u32 {
        unsafe {
            u32::from(self.nrows).extract()
        }
    }
}

impl<T> MatrixCol for Mat<T> {
    fn col(&self, i: u32) -> Col<T> {
        unsafe {
            let v: *const SubMat<T> = &self.slice(..);
            (*v).col(i)
        }
    }
}

impl<T> MatrixColMut for Mat<T> {}

impl<T> MatrixCols for Mat<T> {
    fn cols(&self) -> Cols<T> {
        Cols(self.slice(..))
    }
}

impl<T> MatrixColsMut for Mat<T> {}

impl<T> MatrixDiag for Mat<T> {
    fn diag<'a>(&'a self, i: i32) -> Diag<'a, T> {
        unsafe {
            let v: *const SubMat<T> = &self.slice(..);
            (*v).diag(i)
        }
    }
}

impl<T> MatrixDiagMut for Mat<T> {}

impl<'a, T> MatrixHStripes for Mat<T> {
    fn hstripes(&self, size: u32) -> HStripes<T> {
        assert!(size > 0);

        let size = i32::from(size).unwrap();

        HStripes {
            size: size,
            mat: self.slice(..),
        }
    }
}

impl<'a, T> MatrixHStripesMut for Mat<T> {}

impl<T> MatrixRow for Mat<T> {
    fn row(&self, i: u32) -> Row<T> {
        unsafe {
            let v: *const SubMat<T> = &self.slice(..);
            (*v).row(i)
        }
    }
}

impl<T> MatrixRowMut for Mat<T> {}

impl<T> MatrixRows for Mat<T> {
    fn rows(&self) -> Rows<T> {
        Rows(self.slice(..))
    }
}

impl<T> MatrixRowsMut for Mat<T> {}

impl<'a, T> MatrixVStripes for Mat<T> {
    fn vstripes(&self, size: u32) -> VStripes<T> {
        assert!(size > 0);

        let size = i32::from(size).unwrap();

        VStripes {
            size: size,
            mat: self.slice(..),
        }
    }
}

impl<'a, T> MatrixVStripesMut for Mat<T> {}

impl<'a, T> Slice<'a, RangeFull> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, _: RangeFull) -> SubMat<'a, T> {
        unsafe {
            SubMat::new(*self.data, (self.nrows, self.ncols), self.nrows)
        }
    }
}

impl<'a, T> SliceMut<'a, RangeFull> for Mat<T> {
    type Output = SubMatMut<'a, T>;

    fn slice_mut(&'a mut self, _: RangeFull) -> SubMatMut<'a, T> {
        SubMatMut(self.slice(..))
    }
}

// NOTE Secondary
impl<'a, T> Slice<'a, (u32, RangeFull)> for Mat<T> {
    type Output = Row<'a, T>;

    fn slice(&'a self, (i, _): (u32, RangeFull)) -> Row<'a, T> {
        self.row(i)
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (u32, Range<u32>)> for Mat<T> {
    type Output = Row<'a, T>;

    fn slice(&'a self, (i, r): (u32, Range<u32>)) -> Row<'a, T> {
        self.slice((i, ..)).slice(r)
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (u32, RangeFrom<u32>)> for Mat<T> {
    type Output = Row<'a, T>;

    fn slice(&'a self, (i, r): (u32, RangeFrom<u32>)) -> Row<'a, T> {
        self.slice((i, ..)).slice(r)
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (u32, RangeTo<u32>)> for Mat<T> {
    type Output = Row<'a, T>;

    fn slice(&'a self, (i, r): (u32, RangeTo<u32>)) -> Row<'a, T> {
        self.slice((i, ..)).slice(r)
    }
}

// NOTE Secondary
impl<'a, T> Slice<'a, (RangeFull, u32)> for Mat<T> {
    type Output = Col<'a, T>;

    fn slice(&'a self, (_, i): (RangeFull, u32)) -> Col<'a, T> {
        self.col(i)
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (Range<u32>, u32)> for Mat<T> {
    type Output = Col<'a, T>;

    fn slice(&'a self, (r, i): (Range<u32>, u32)) -> Col<'a, T> {
        self.slice((.., i)).slice(r)
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeFrom<u32>, u32)> for Mat<T> {
    type Output = Col<'a, T>;

    fn slice(&'a self, (r, i): (RangeFrom<u32>, u32)) -> Col<'a, T> {
        self.slice((.., i)).slice(r)
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeTo<u32>, u32)> for Mat<T> {
    type Output = Col<'a, T>;

    fn slice(&'a self, (r, i): (RangeTo<u32>, u32)) -> Col<'a, T> {
        self.slice((.., i)).slice(r)
    }
}

// NOTE Secondary
impl<'a, T> Slice<'a, (Range<u32>, Range<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, r: (Range<u32>, Range<u32>)) -> SubMat<'a, T> {
        // HACK use raw pointers to work-around the lack of re-borrow semantics
        unsafe {
            let v: *const SubMat<T> = &self.slice(..);
            (*v).slice(r)
        }
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeFull, Range<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (_, c): (RangeFull, Range<u32>)) -> SubMat<'a, T> {
        self.slice((0..self.nrows(), c))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeFull, RangeFrom<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (_, c): (RangeFull, RangeFrom<u32>)) -> SubMat<'a, T> {
        self.slice((0..self.nrows(), c))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeFull, RangeTo<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (_, c): (RangeFull, RangeTo<u32>)) -> SubMat<'a, T> {
        self.slice((0..self.nrows(), c))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (Range<u32>, RangeFull)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, _): (Range<u32>, RangeFull)) -> SubMat<'a, T> {
        self.slice((r, 0..self.ncols()))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (Range<u32>, RangeFrom<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, c): (Range<u32>, RangeFrom<u32>)) -> SubMat<'a, T> {
        self.slice((r, c.start..self.ncols()))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (Range<u32>, RangeTo<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, c): (Range<u32>, RangeTo<u32>)) -> SubMat<'a, T> {
        self.slice((r, 0..c.end))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeFrom<u32>, Range<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, c): (RangeFrom<u32>, Range<u32>)) -> SubMat<'a, T> {
        self.slice((r.start..self.nrows(), c))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeFrom<u32>, RangeFull)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, _): (RangeFrom<u32>, RangeFull)) -> SubMat<'a, T> {
        self.slice((r.start..self.nrows(), 0..self.ncols()))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeFrom<u32>, RangeFrom<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, c): (RangeFrom<u32>, RangeFrom<u32>)) -> SubMat<'a, T> {
        self.slice((r.start..self.nrows(), c.start..self.ncols()))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeFrom<u32>, RangeTo<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, c): (RangeFrom<u32>, RangeTo<u32>)) -> SubMat<'a, T> {
        self.slice((r.start..self.nrows(), 0..c.end))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeTo<u32>, Range<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, c): (RangeTo<u32>, Range<u32>)) -> SubMat<'a, T> {
        self.slice((0..r.end, c))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeTo<u32>, RangeFull)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, _): (RangeTo<u32>, RangeFull)) -> SubMat<'a, T> {
        self.slice((0..r.end, 0..self.ncols()))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeTo<u32>, RangeFrom<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, c): (RangeTo<u32>, RangeFrom<u32>)) -> SubMat<'a, T> {
        self.slice((0..r.end, c.start..self.ncols()))
    }
}

// NOTE Forward
impl<'a, T> Slice<'a, (RangeTo<u32>, RangeTo<u32>)> for Mat<T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, (r, c): (RangeTo<u32>, RangeTo<u32>)) -> SubMat<'a, T> {
        self.slice((0..r.end, 0..c.end))
    }
}

macro_rules! forward {
    ($ty:ident { $(($r:ty, $c:ty)),+, }) => {
        $(
            // NOTE Forward
            impl<'a, T> SliceMut<'a, ($r, $c)> for Mat<T> {
                type Output = $ty<'a, T>;

                fn slice_mut(&'a mut self, r: ($r, $c)) -> $ty<'a, T> {
                    $ty(self.slice(r))
                }
            }
         )+
    }
}

forward!(ColMut {
    (Range<u32>, u32),
    (RangeFrom<u32>, u32),
    (RangeFull, u32),
    (RangeTo<u32>, u32),
});

forward!(RowMut {
    (u32, Range<u32>),
    (u32, RangeFrom<u32>),
    (u32, RangeFull),
    (u32, RangeTo<u32>),
});

forward!(SubMatMut {
    (Range<u32>, Range<u32>),
    (Range<u32>, RangeFrom<u32>),
    (Range<u32>, RangeFull),
    (Range<u32>, RangeTo<u32>),
    (RangeFrom<u32>, Range<u32>),
    (RangeFrom<u32>, RangeFrom<u32>),
    (RangeFrom<u32>, RangeFull),
    (RangeFrom<u32>, RangeTo<u32>),
    (RangeFull, Range<u32>),
    (RangeFull, RangeFrom<u32>),
    (RangeFull, RangeTo<u32>),
    (RangeTo<u32>, Range<u32>),
    (RangeTo<u32>, RangeFrom<u32>),
    (RangeTo<u32>, RangeFull),
    (RangeTo<u32>, RangeTo<u32>),
});

impl<T> Transpose for Mat<T> {
    type Output = Transposed<Mat<T>>;

    fn t(self) -> Transposed<Mat<T>> {
        Transposed(self)
    }
}

impl<'a, T> Transpose for &'a Mat<T> {
    type Output = Transposed<SubMat<'a, T>>;

    fn t(self) -> Transposed<SubMat<'a, T>> {
        self.slice(..).t()
    }
}

impl<'a, T> Transpose for &'a mut Mat<T> {
    type Output = Transposed<SubMatMut<'a, T>>;

    fn t(self) -> Transposed<SubMatMut<'a, T>> {
        self.slice_mut(..).t()
    }
}

impl<T> VSplit for Mat<T> {
    fn vsplit_at(&self, i: u32) -> (SubMat<T>, SubMat<T>) {
        unsafe {
            let v: *const SubMat<T> = &self.slice(..);
            (*v).vsplit_at(i)
        }
    }
}

impl<T> VSplitMut for Mat<T> {
    fn vsplit_at_mut(&mut self, i: u32) -> (SubMatMut<T>, SubMatMut<T>) {
        unsafe {
            let v: *mut SubMatMut<T> = &mut self.slice_mut(..);
            (*v).vsplit_at_mut(i)
        }
    }
}
