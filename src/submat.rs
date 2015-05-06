//! Sub-matrix views

use core::nonzero::NonZero;
use std::cmp;
use std::iter::IntoIterator;
use std::ops::{Index, Range, RangeFrom, RangeFull, RangeTo};

use cast::From;
use extract::Extract;

use traits::{
    HSplit, Matrix, MatrixCol, MatrixCols, MatrixDiag, MatrixHStripes, MatrixRow, MatrixRows,
    MatrixVStripes, Slice, Transpose, VSplit, self,
};
use {Col, Cols, Diag, HStripes, Row, Rows, Transposed, VStripes, SubMat, SubMatMut};

/// An iterator over an immutable sub-matrix view
pub struct Iter<'a, T> {
    mat: SubMat<'a, T>,
    row: i32,
}

impl<'a, T> Iter<'a, T> {
    fn new(mat: SubMat<'a, T>) -> Iter<'a, T> {
        Iter {
            mat: if mat.nrows == 0 { SubMat { ncols: 0, ..mat } } else { mat },
            row: 0,
        }
    }

    unsafe fn raw_next(&mut self) -> Option<NonZero<*mut T>> {
        if self.mat.ncols == 0 {
            None
        } else {
            let ptr = self.mat.unsafe_index((self.row, 0));

            self.row += 1;

            if self.row == self.mat.nrows {
                self.row = 0;
                let v: *const SubMat<T> = &self.mat;
                self.mat = (*v).unsafe_slice((0, 1)..(self.mat.nrows, self.mat.ncols));
            }

            Some(NonZero::new(ptr))
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        unsafe {
            self.raw_next().map(|x| &**x)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        unsafe {
            let nrows = usize::from(self.mat.nrows).extract();
            let ncols = usize::from(self.mat.ncols).extract();
            let total = nrows * ncols;
            let done = usize::from(self.row).extract();
            let exact = total - done;

            (exact, Some(exact))
        }
    }
}

/// An iterator over a mutable sub-matrix "view"
pub struct IterMut<'a, T>(Iter<'a, T>);

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        unsafe {
            self.0.raw_next().map(|x| &mut **x)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, T> HSplit for SubMat<'a, T> {
    fn hsplit_at(&self, i: u32) -> (SubMat<T> , SubMat<T>) {
        unsafe {
            assert!(i <= self.nrows());

            self.unsafe_hsplit_at(i32::from(i).extract())
        }
    }
}

impl<'a, T> Index<(u32, u32)> for SubMat<'a, T> {
    type Output = T;

    fn index(&self, (row, col): (u32, u32)) -> &T {
        unsafe {
            &*self.raw_index((row, col))
        }
    }
}

impl<'a, T> IntoIterator for SubMat<'a, T> {
    type IntoIter = Iter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Iter<'a, T> {
        Iter::new(self)
    }
}

impl<'a, 'b, T> traits::Iter<'a> for SubMat<'b, T> {
    type Iter = Iter<'b, T>;

    fn iter(&'a self) -> Iter<'b, T> {
        self.into_iter()
    }
}

impl<'a, T> Matrix for SubMat<'a, T> {
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

// NOTE Forward
impl<'a, T> MatrixCol for SubMat<'a, T> {
    fn col(&self, i: u32) -> Col<T> {
        self.slice((.., i))
    }
}

impl<'a, T> MatrixCols for SubMat<'a, T> {
    fn cols(&self) -> Cols<T> {
        Cols(*self)
    }
}

impl<'a, T> MatrixDiag for SubMat<'a, T> {
    fn diag(&self, i: i32) -> Diag<T> {
        unsafe {
            use Slice;

            let (nrows, ncols) = (self.nrows, self.ncols);
            let stride = self.stride;

            if i > 0 {
                assert!(i < ncols);

                let data = self.data.offset(isize::from(i) * isize::from(stride));
                let len = cmp::min(nrows, ncols - i);

                Diag(Slice::new(data, len, stride + 1))
            } else {
                let i = -i;

                assert!(i < nrows);

                let data = self.data.offset(isize::from(i));
                let len = cmp::min(nrows - i, ncols);

                Diag(Slice::new(data, len, stride + 1))
            }
        }
    }
}

impl<'a, T> MatrixHStripes for SubMat<'a, T> {
    fn hstripes(&self, size: u32) -> HStripes<T> {
        assert!(size > 0);

        let size = i32::from(size).unwrap();

        HStripes {
            mat: *self,
            size: size,
        }
    }
}

impl<'a, T> MatrixRow for SubMat<'a, T> {
    fn row(&self, i: u32) -> Row<T> {
        self.slice((i, ..))
    }
}

impl<'a, T> MatrixRows for SubMat<'a, T> {
    fn rows(&self) -> Rows<T> {
        Rows(*self)
    }
}

impl<'a, T> MatrixVStripes for SubMat<'a, T> {
    fn vstripes(&self, size: u32) -> VStripes<T> {
        assert!(size > 0);

        let size = i32::from(size).unwrap();

        VStripes {
            size: size,
            mat: *self,
        }
    }
}

impl<'a, 'b, T> Slice<'a, RangeFull> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, _: RangeFull) -> SubMat<'b, T> {
        *self
    }
}

// NOTE Core
impl<'a, 'b, T> Slice<'a, (u32, RangeFull)> for SubMat<'b, T> {
    type Output = Row<'b, T>;

    fn slice(&'a self, (i, _): (u32, RangeFull)) -> Row<'b, T> {
        unsafe {
            assert!(i < self.nrows());

            self.unsafe_row(i32::from(i).extract())
        }
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (u32, Range<u32>)> for SubMat<'b, T> {
    type Output = Row<'b, T>;

    fn slice(&'a self, (i, r): (u32, Range<u32>)) -> Row<'b, T> {
        self.slice((i, ..)).slice(r)
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (u32, RangeFrom<u32>)> for SubMat<'b, T> {
    type Output = Row<'b, T>;

    fn slice(&'a self, (i, r): (u32, RangeFrom<u32>)) -> Row<'b, T> {
        self.slice((i, ..)).slice(r)
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (u32, RangeTo<u32>)> for SubMat<'b, T> {
    type Output = Row<'b, T>;

    fn slice(&'a self, (i, r): (u32, RangeTo<u32>)) -> Row<'b, T> {
        self.slice((i, ..)).slice(r)
    }
}

// NOTE Core
impl<'a, 'b, T> Slice<'a, (RangeFull, u32)> for SubMat<'b, T> {
    type Output = Col<'b, T>;

    fn slice(&'a self, (_, i): (RangeFull, u32)) -> Col<'b, T> {
        unsafe {
            assert!(i < self.ncols());

            self.unsafe_col(i32::from(i).extract())
        }
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (Range<u32>, u32)> for SubMat<'b, T> {
    type Output = Col<'b, T>;

    fn slice(&'a self, (r, i): (Range<u32>, u32)) -> Col<'b, T> {
        self.slice((.., i)).slice(r)
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeFrom<u32>, u32)> for SubMat<'b, T> {
    type Output = Col<'b, T>;

    fn slice(&'a self, (r, i): (RangeFrom<u32>, u32)) -> Col<'b, T> {
        self.slice((.., i)).slice(r)
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeTo<u32>, u32)> for SubMat<'b, T> {
    type Output = Col<'b, T>;

    fn slice(&'a self, (r, i): (RangeTo<u32>, u32)) -> Col<'b, T> {
        self.slice((.., i)).slice(r)
    }
}

// NOTE Core
impl<'a, 'b, T> Slice<'a, (Range<u32>, Range<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, c): (Range<u32>, Range<u32>)) -> SubMat<'b, T> {
        unsafe {
            let (nrows, ncols) = self.size();
            let Range { start: scol, end: ecol } = c;
            let Range { start: srow, end: erow } = r;

            assert!(srow <= erow && erow <= nrows && scol <= ecol && ecol <= ncols);

            let start = (i32::from(srow).extract(), i32::from(scol).extract());
            let end = (i32::from(erow).extract(), i32::from(ecol).extract());

            self.unsafe_slice(start..end)
        }
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeFull, Range<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (_, c): (RangeFull, Range<u32>)) -> SubMat<'b, T> {
        self.slice((0..self.nrows(), c))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeFull, RangeFrom<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (_, c): (RangeFull, RangeFrom<u32>)) -> SubMat<'b, T> {
        self.slice((0..self.nrows(), c))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeFull, RangeTo<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (_, c): (RangeFull, RangeTo<u32>)) -> SubMat<'b, T> {
        self.slice((0..self.nrows(), c))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (Range<u32>, RangeFull)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, _): (Range<u32>, RangeFull)) -> SubMat<'b, T> {
        self.slice((r, 0..self.ncols()))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (Range<u32>, RangeFrom<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, c): (Range<u32>, RangeFrom<u32>)) -> SubMat<'b, T> {
        self.slice((r, c.start..self.ncols()))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (Range<u32>, RangeTo<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, c): (Range<u32>, RangeTo<u32>)) -> SubMat<'b, T> {
        self.slice((r, 0..c.end))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeFrom<u32>, Range<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, c): (RangeFrom<u32>, Range<u32>)) -> SubMat<'b, T> {
        self.slice((r.start..self.nrows(), c))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeFrom<u32>, RangeFull)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, _): (RangeFrom<u32>, RangeFull)) -> SubMat<'b, T> {
        self.slice((r.start..self.nrows(), 0..self.ncols()))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeFrom<u32>, RangeFrom<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, c): (RangeFrom<u32>, RangeFrom<u32>)) -> SubMat<'b, T> {
        self.slice((r.start..self.nrows(), c.start..self.ncols()))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeFrom<u32>, RangeTo<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, c): (RangeFrom<u32>, RangeTo<u32>)) -> SubMat<'b, T> {
        self.slice((r.start..self.nrows(), 0..c.end))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeTo<u32>, Range<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, c): (RangeTo<u32>, Range<u32>)) -> SubMat<'b, T> {
        self.slice((0..r.end, c))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeTo<u32>, RangeFull)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, _): (RangeTo<u32>, RangeFull)) -> SubMat<'b, T> {
        self.slice((0..r.end, 0..self.ncols()))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeTo<u32>, RangeFrom<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, c): (RangeTo<u32>, RangeFrom<u32>)) -> SubMat<'b, T> {
        self.slice((0..r.end, c.start..self.ncols()))
    }
}

// NOTE Forward
impl<'a, 'b, T> Slice<'a, (RangeTo<u32>, RangeTo<u32>)> for SubMat<'b, T> {
    type Output = SubMat<'b, T>;

    fn slice(&'a self, (r, c): (RangeTo<u32>, RangeTo<u32>)) -> SubMat<'b, T> {
        self.slice((0..r.end, 0..c.end))
    }
}

impl<'a, T> Transpose for SubMat<'a, T> {
    type Output = Transposed<SubMat<'a, T>>;

    fn t(self) -> Transposed<SubMat<'a, T>> {
        Transposed(self)
    }
}

impl<'a, T> VSplit for SubMat<'a, T> {
    fn vsplit_at(&self, i: u32) -> (SubMat<T> , SubMat<T>) {
        unsafe {
            assert!(i <= self.ncols());

            self.unsafe_vsplit_at(i32::from(i).extract())
        }
    }
}

// NB All `impl`s below this point *shouldn't* be in this module, but intra-crate privacy won't
// let me move these `impl`s anywhere else

impl<'a, T> IntoIterator for SubMatMut<'a, T> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> IterMut<'a, T> {
        IterMut(self.0.into_iter())
    }
}

impl<'a, 'b, T> traits::IterMut<'a> for SubMatMut<'b, T> {
    type IterMut = IterMut<'a, T>;

    fn iter_mut(&'a mut self) -> IterMut<'a, T> {
        IterMut(self.0.into_iter())
    }
}
