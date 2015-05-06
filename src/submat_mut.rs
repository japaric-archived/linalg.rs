use std::iter::IntoIterator;
use std::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo};

use traits::{
    HSplit, HSplitMut, Matrix, MatrixCol, MatrixColMut, MatrixCols, MatrixColsMut, MatrixDiag,
    MatrixDiagMut, MatrixHStripes, MatrixHStripesMut, MatrixRow, MatrixRowMut, MatrixRows,
    MatrixRowsMut, MatrixVStripes, MatrixVStripesMut, Slice, SliceMut, Transpose, VSplit,
    VSplitMut, self
};
use submat::{Iter, IterMut};
use {
    Col, ColMut, Cols, Diag, HStripes, Row, RowMut, Rows, Transposed, VStripes, SubMat, SubMatMut,
};

impl<'a, T> HSplit for SubMatMut<'a, T> {
    fn hsplit_at(&self, i: u32) -> (SubMat<T>, SubMat<T>) {
        self.0.hsplit_at(i)
    }
}

impl<'a, T> HSplitMut for SubMatMut<'a, T> {
    fn hsplit_at_mut(&mut self, i: u32) -> (SubMatMut<T>, SubMatMut<T>) {
        let (left, right) = self.0.hsplit_at(i);

        (SubMatMut(left), SubMatMut(right))
    }
}

impl<'a, T> Index<(u32, u32)> for SubMatMut<'a, T> {
    type Output = T;

    fn index(&self, (row, col): (u32, u32)) -> &T {
        &self.0[(row, col)]
    }
}

impl<'a, T> IndexMut<(u32, u32)> for SubMatMut<'a, T> {
    fn index_mut(&mut self, (row, col): (u32, u32)) -> &mut T {
        unsafe {
            &mut *self.0.raw_index((row, col))
        }
    }
}

impl<'a, 'b, T> IntoIterator for &'a SubMatMut<'b, T> {
    type IntoIter = Iter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Iter<'a, T> {
        use traits::Iter;

        self.iter()
    }
}

impl<'a, 'b, T> IntoIterator for &'a mut SubMatMut<'b, T> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> IterMut<'a, T> {
        use traits::IterMut;

        self.iter_mut()
    }
}

impl<'a, 'b, T> traits::Iter<'a> for SubMatMut<'b, T> {
    type Iter = Iter<'a, T>;

    fn iter(&'a self) -> Iter<'a, T> {
        self.0.iter()
    }
}

impl<'a, T> Matrix for SubMatMut<'a, T> {
    type Elem = T;

    fn ncols(&self) -> u32 {
        self.0.ncols()
    }

    fn nrows(&self) -> u32 {
        self.0.nrows()
    }

    fn size(&self) -> (u32, u32) {
        self.0.size()
    }
}

impl<'a, T> MatrixCol for SubMatMut<'a, T> {
    fn col(&self, i: u32) -> Col<T> {
        self.0.col(i)
    }
}

impl<'a, T> MatrixColMut for SubMatMut<'a, T> {}

impl<'a, T> MatrixCols for SubMatMut<'a, T> {
    fn cols(&self) -> Cols<T> {
        self.0.cols()
    }
}

impl<'a, T> MatrixColsMut for SubMatMut<'a, T> {}

impl<'a, T> MatrixDiag for SubMatMut<'a, T> {
    fn diag(&self, i: i32) -> Diag<T> {
        self.0.diag(i)
    }
}

impl<'a, T> MatrixDiagMut for SubMatMut<'a, T> {}

impl<'a, T> MatrixHStripes for SubMatMut<'a, T> {
    fn hstripes(&self, size: u32) -> HStripes<T> {
        self.0.hstripes(size)
    }
}

impl<'a, T> MatrixHStripesMut for SubMatMut<'a, T> {}

impl<'a, T> MatrixRow for SubMatMut<'a, T> {
    fn row(&self, i: u32) -> Row<T> {
        self.0.row(i)
    }
}

impl<'a, T> MatrixRowMut for SubMatMut<'a, T> {}

impl<'a, T> MatrixRows for SubMatMut<'a, T> {
    fn rows(&self) -> Rows<T> {
        self.0.rows()
    }
}

impl<'a, T> MatrixRowsMut for SubMatMut<'a, T> {}

impl<'a, T> MatrixVStripes for SubMatMut<'a, T> {
    fn vstripes(&self, size: u32) -> VStripes<T> {
        self.0.vstripes(size)
    }
}

impl<'a, T> MatrixVStripesMut for SubMatMut<'a, T> {}

impl<'a, 'b, T> Slice<'a, RangeFull> for SubMatMut<'b, T> {
    type Output = SubMat<'a, T>;

    fn slice(&'a self, _: RangeFull) -> SubMat<'a, T> {
        self.0
    }
}

impl<'a, 'b, T> SliceMut<'a, RangeFull> for SubMatMut<'b, T> {
    type Output = SubMatMut<'a, T>;

    fn slice_mut(&'a mut self, _: RangeFull) -> SubMatMut<'a, T> {
        SubMatMut(self.0)
    }
}

macro_rules! forward {
    ($ty:ident, $ty_mut:ident { $(($r:ty, $c:ty)),+, }  ) => {
        $(
            // NOTE Forward
            impl<'a, 'b, T> Slice<'a, ($r, $c)> for SubMatMut<'b, T> {
                type Output = $ty<'a, T>;

                fn slice(&'a self, r: ($r, $c)) -> $ty<'a, T> {
                    self.0.slice(r)
                }
            }

            // NOTE Forward
            impl<'a, 'b, T> SliceMut<'a, ($r, $c)> for SubMatMut<'b, T> {
                type Output = $ty_mut<'a, T>;

                fn slice_mut(&'a mut self, r: ($r, $c)) -> $ty_mut<'a, T> {
                    $ty_mut(self.0.slice(r))
                }
            }
         )+
    }
}

forward!(Col, ColMut {
    (Range<u32>, u32),
    (RangeFrom<u32>, u32),
    (RangeFull, u32),
    (RangeTo<u32>, u32),
});

forward!(Row, RowMut {
    (u32, Range<u32>),
    (u32, RangeFrom<u32>),
    (u32, RangeFull),
    (u32, RangeTo<u32>),
});

forward!(SubMat, SubMatMut {
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

impl<'a, T> Transpose for SubMatMut<'a, T> {
    type Output = Transposed<SubMatMut<'a, T>>;

    fn t(self) -> Transposed<SubMatMut<'a, T>> {
        Transposed(self)
    }
}

impl<'a, T> VSplit for SubMatMut<'a, T> {
    fn vsplit_at(&self, i: u32) -> (SubMat<T>, SubMat<T>) {
        self.0.vsplit_at(i)
    }
}

impl<'a, T> VSplitMut for SubMatMut<'a, T> {
    fn vsplit_at_mut(&mut self, i: u32) -> (SubMatMut<T>, SubMatMut<T>) {
        let (left, right) = self.0.vsplit_at(i);

        (SubMatMut(left), SubMatMut(right))
    }
}
