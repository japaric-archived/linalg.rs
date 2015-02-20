use std::mem;
use std::raw::{self, Repr};
use std::ops::{Range, RangeFrom, RangeTo};

use {
    Col, ColVec, Diag, Error, Mat, MutCol, MutDiag, MutRow, MutView, Result, Row, RowVec, View,
    strided,
};
use traits::{Matrix, Slice, SliceMut};

impl<T> ::Slice for [T] {
    type Ty = T;

    fn slice(&self, Range { start, end }: Range<usize>) -> Result<strided::Slice<T>> {
        let raw::Slice { data, len } = self.repr();

        if start > end {
            Err(Error::InvalidSlice)
        } else if end > len {
            Err(Error::OutOfBounds)
        } else {
            Ok(unsafe { ::From::parts((
                data.offset(start as isize),
                end - start,
                1,
            ))})
        }
    }
}

impl<'a, T> Slice<'a, Range<usize>> for ColVec<T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, range: Range<usize>) -> Result<Col<'a, T>> {
        ::Slice::slice(&*self.0, range).map(Col)
    }
}

impl<'a, T> Slice<'a, RangeFrom<usize>> for ColVec<T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, range: RangeFrom<usize>) -> Result<Col<'a, T>> {
        self.slice(range.start..self.len())
    }
}

impl<'a, T> Slice<'a, RangeTo<usize>> for ColVec<T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, range: RangeTo<usize>) -> Result<Col<'a, T>> {
        self.slice(0..range.end)
    }
}

impl<'a, 'b, T> Slice<'a, Range<usize>> for Col<'b, T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, range: Range<usize>) -> Result<Col<'a, T>> {
        self.0.slice(range).map(Col)
    }
}

impl<'a, 'b, T> Slice<'a, RangeFrom<usize>> for Col<'b, T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, range: RangeFrom<usize>) -> Result<Col<'a, T>> {
        self.slice(range.start..self.len())
    }
}

impl<'a, 'b, T> Slice<'a, RangeTo<usize>> for Col<'b, T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, range: RangeTo<usize>) -> Result<Col<'a, T>> {
        self.slice(0..range.end)
    }
}

impl<'a, 'b, T> Slice<'a, Range<usize>> for Diag<'b, T> {
    type Slice = Diag<'a, T>;

    fn slice(&'a self, range: Range<usize>) -> Result<Diag<'a, T>> {
        self.0.slice(range).map(Diag)
    }
}

impl<'a, 'b, T> Slice<'a, RangeFrom<usize>> for Diag<'b, T> {
    type Slice = Diag<'a, T>;

    fn slice(&'a self, range: RangeFrom<usize>) -> Result<Diag<'a, T>> {
        self.0.slice(range.start..self.len()).map(Diag)
    }
}

impl<'a, 'b, T> Slice<'a, RangeTo<usize>> for Diag<'b, T> {
    type Slice = Diag<'a, T>;

    fn slice(&'a self, range: RangeTo<usize>) -> Result<Diag<'a, T>> {
        self.0.slice(0..range.end).map(Diag)
    }
}

impl<'a, T> Slice<'a, Range<(usize, usize)>> for Mat<T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, Range { start, end }: Range<(usize, usize)>) -> Result<View<'a, T>> {
        let (start_row, start_col) = start;
        let (end_row, end_col) = end;
        let (nrows, ncols) = self.size();

        if end_col > ncols || end_row > nrows {
            Err(Error::OutOfBounds)
        } else if start_col > end_col || start_row > end_row {
            Err(Error::InvalidSlice)
        } else {
            let stride = nrows;
            let ptr = unsafe {
                self.data.as_ptr().offset((start_col * stride + start_row) as isize)
            };

            Ok(unsafe { ::From::parts((
                ptr,
                end_row - start_row,
                end_col - start_col,
                stride,
            ))})
        }
    }
}

impl<'a, T> Slice<'a, RangeFrom<(usize, usize)>> for Mat<T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, range: RangeFrom<(usize, usize)>) -> Result<View<'a, T>> {
        self.slice(range.start..self.size())
    }
}

impl<'a, T> Slice<'a, RangeTo<(usize, usize)>> for Mat<T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, range: RangeTo<(usize, usize)>) -> Result<View<'a, T>> {
        self.slice((0, 0)..range.end)
    }
}

impl<'a, 'b, T> Slice<'a, Range<usize>> for MutCol<'b, T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, range: Range<usize>) -> Result<Col<'a, T>> {
        self.0.slice(range).map(Col)
    }
}

impl<'a, 'b, T> Slice<'a, RangeFrom<usize>> for MutCol<'b, T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, range: RangeFrom<usize>) -> Result<Col<'a, T>> {
        self.slice(range.start..self.len())
    }
}

impl<'a, 'b, T> Slice<'a, RangeTo<usize>> for MutCol<'b, T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, range: RangeTo<usize>) -> Result<Col<'a, T>> {
        self.slice(0..range.end)
    }
}

impl<'a, 'b, T> Slice<'a, Range<usize>> for MutDiag<'b, T> {
    type Slice = Diag<'a, T>;

    fn slice(&'a self, range: Range<usize>) -> Result<Diag<'a, T>> {
        self.0.slice(range).map(Diag)
    }
}

impl<'a, 'b, T> Slice<'a, RangeFrom<usize>> for MutDiag<'b, T> {
    type Slice = Diag<'a, T>;

    fn slice(&'a self, range: RangeFrom<usize>) -> Result<Diag<'a, T>> {
        self.slice(range.start..self.len())
    }
}

impl<'a, 'b, T> Slice<'a, RangeTo<usize>> for MutDiag<'b, T> {
    type Slice = Diag<'a, T>;

    fn slice(&'a self, range: RangeTo<usize>) -> Result<Diag<'a, T>> {
        self.slice(0..range.end)
    }
}

impl<'a, 'b, T> Slice<'a, Range<usize>> for MutRow<'b, T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, range: Range<usize>) -> Result<Row<'a, T>> {
        self.0.slice(range).map(Row)
    }
}

impl<'a, 'b, T> Slice<'a, RangeFrom<usize>> for MutRow<'b, T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, range: RangeFrom<usize>) -> Result<Row<'a, T>> {
        self.slice(range.start..self.len())
    }
}

impl<'a, 'b, T> Slice<'a, RangeTo<usize>> for MutRow<'b, T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, range: RangeTo<usize>) -> Result<Row<'a, T>> {
        self.slice(0..range.end)
    }
}

impl<'a, T> Slice<'a, Range<usize>> for RowVec<T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, range: Range<usize>) -> Result<Row<'a, T>> {
        ::Slice::slice(&*self.0, range).map(Row)
    }
}

impl<'a, T> Slice<'a, RangeFrom<usize>> for RowVec<T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, range: RangeFrom<usize>) -> Result<Row<'a, T>> {
        self.slice(range.start..self.len())
    }
}

impl<'a, T> Slice<'a, RangeTo<usize>> for RowVec<T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, range: RangeTo<usize>) -> Result<Row<'a, T>> {
        self.slice(0..range.end)
    }
}

impl<'a, 'b, T> Slice<'a, Range<usize>> for Row<'b, T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, range: Range<usize>) -> Result<Row<'a, T>> {
        self.0.slice(range).map(Row)
    }
}

impl<'a, 'b, T> Slice<'a, RangeFrom<usize>> for Row<'b, T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, range: RangeFrom<usize>) -> Result<Row<'a, T>> {
        self.slice(range.start..self.len())
    }
}

impl<'a, 'b, T> Slice<'a, RangeTo<usize>> for Row<'b, T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, range: RangeTo<usize>) -> Result<Row<'a, T>> {
        self.slice(0..range.end)
    }
}

impl<'a, 'b, T> Slice<'a, Range<(usize, usize)>> for MutView<'b, T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, range: Range<(usize, usize)>) -> Result<View<'a, T>> {
        unsafe {
            mem::transmute(self.0.slice(range))
        }
    }
}

impl<'a, 'b, T> Slice<'a, RangeFrom<(usize, usize)>> for MutView<'b, T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, range: RangeFrom<(usize, usize)>) -> Result<View<'a, T>> {
        self.slice(range.start..self.size())
    }
}

impl<'a, 'b, T> Slice<'a, RangeTo<(usize, usize)>> for MutView<'b, T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, range: RangeTo<(usize, usize)>) -> Result<View<'a, T>> {
        self.slice((0, 0)..range.end)
    }
}

impl<'a, 'b, T> Slice<'a, Range<(usize, usize)>> for View<'b, T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, range: Range<(usize, usize)>) -> Result<View<'a, T>> {
        unsafe {
            mem::transmute(self.0.slice(range))
        }
    }
}

impl<'a, 'b, T> Slice<'a, RangeFrom<(usize, usize)>> for View<'b, T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, range: RangeFrom<(usize, usize)>) -> Result<View<'a, T>> {
        self.slice(range.start..self.size())
    }
}

impl<'a, 'b, T> Slice<'a, RangeTo<(usize, usize)>> for View<'b, T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, range: RangeTo<(usize, usize)>) -> Result<View<'a, T>> {
        self.slice((0, 0)..range.end)
    }
}

impl<'a, T> SliceMut<'a, Range<usize>> for ColVec<T> {
    type Slice = MutCol<'a, T>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Result<MutCol<'a, T>> {
        unsafe {
            mem::transmute(::Slice::slice(&*self.0, range))
        }
    }
}

impl<'a, T> SliceMut<'a, RangeFrom<usize>> for ColVec<T> {
    type Slice = MutCol<'a, T>;

    fn slice_mut(&'a mut self, range: RangeFrom<usize>) -> Result<MutCol<'a, T>> {
        let n = self.len();
        self.slice_mut(range.start..n)
    }
}

impl<'a, T> SliceMut<'a, RangeTo<usize>> for ColVec<T> {
    type Slice = MutCol<'a, T>;

    fn slice_mut(&'a mut self, range: RangeTo<usize>) -> Result<MutCol<'a, T>> {
        self.slice_mut(0..range.end)
    }
}

impl<'a, T> SliceMut<'a, Range<(usize, usize)>> for Mat<T> {
    type Slice = MutView<'a, T>;

    fn slice_mut(&'a mut self, range: Range<(usize, usize)>) -> Result<MutView<'a, T>> {
        unsafe {
            mem::transmute(self.slice(range))
        }
    }
}

impl<'a, T> SliceMut<'a, RangeFrom<(usize, usize)>> for Mat<T> {
    type Slice = MutView<'a, T>;

    fn slice_mut(&'a mut self, range: RangeFrom<(usize, usize)>) -> Result<MutView<'a, T>> {
        let sz = self.size();
        self.slice_mut(range.start..sz)
    }
}

impl<'a, T> SliceMut<'a, RangeTo<(usize, usize)>> for Mat<T> {
    type Slice = MutView<'a, T>;

    fn slice_mut(&'a mut self, range: RangeTo<(usize, usize)>) -> Result<MutView<'a, T>> {
        self.slice_mut((0, 0)..range.end)
    }
}

impl<'a, 'b, T> SliceMut<'a, Range<usize>> for MutCol<'b, T> {
    type Slice = MutCol<'a, T>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Result<MutCol<'a, T>> {
        self.0.slice_mut(range).map(MutCol)
    }
}

impl<'a, 'b, T> SliceMut<'a, RangeFrom<usize>> for MutCol<'b, T> {
    type Slice = MutCol<'a, T>;

    fn slice_mut(&'a mut self, range: RangeFrom<usize>) -> Result<MutCol<'a, T>> {
        let n = self.len();
        self.slice_mut(range.start..n)
    }
}

impl<'a, 'b, T> SliceMut<'a, RangeTo<usize>> for MutCol<'b, T> {
    type Slice = MutCol<'a, T>;

    fn slice_mut(&'a mut self, range: RangeTo<usize>) -> Result<MutCol<'a, T>> {
        self.slice_mut(0..range.end)
    }
}

impl<'a, 'b, T> SliceMut<'a, Range<usize>> for MutDiag<'b, T> {
    type Slice = MutDiag<'a, T>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Result<MutDiag<'a, T>> {
        self.0.slice_mut(range).map(MutDiag)
    }
}

impl<'a, 'b, T> SliceMut<'a, RangeFrom<usize>> for MutDiag<'b, T> {
    type Slice = MutDiag<'a, T>;

    fn slice_mut(&'a mut self, range: RangeFrom<usize>) -> Result<MutDiag<'a, T>> {
        let n = self.len();
        self.slice_mut(range.start..n)
    }
}

impl<'a, 'b, T> SliceMut<'a, RangeTo<usize>> for MutDiag<'b, T> {
    type Slice = MutDiag<'a, T>;

    fn slice_mut(&'a mut self, range: RangeTo<usize>) -> Result<MutDiag<'a, T>> {
        self.slice_mut(0..range.end)
    }
}

impl<'a, 'b, T> SliceMut<'a, Range<usize>> for MutRow<'b, T> {
    type Slice = MutRow<'a, T>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Result<MutRow<'a, T>> {
        self.0.slice_mut(range).map(MutRow)
    }
}

impl<'a, 'b, T> SliceMut<'a, RangeFrom<usize>> for MutRow<'b, T> {
    type Slice = MutRow<'a, T>;

    fn slice_mut(&'a mut self, range: RangeFrom<usize>) -> Result<MutRow<'a, T>> {
        let n = self.len();
        self.slice_mut(range.start..n)
    }
}

impl<'a, 'b, T> SliceMut<'a, RangeTo<usize>> for MutRow<'b, T> {
    type Slice = MutRow<'a, T>;

    fn slice_mut(&'a mut self, range: RangeTo<usize>) -> Result<MutRow<'a, T>> {
        self.slice_mut(0..range.end)
    }
}

impl<'a, 'b, T> SliceMut<'a, Range<(usize, usize)>> for MutView<'b, T> {
    type Slice = MutView<'a, T>;

    fn slice_mut(&'a mut self, range: Range<(usize, usize)>) -> Result<MutView<'a, T>> {
        unsafe {
            mem::transmute(self.0.slice(range))
        }
    }
}

impl<'a, 'b, T> SliceMut<'a, RangeFrom<(usize, usize)>> for MutView<'b, T> {
    type Slice = MutView<'a, T>;

    fn slice_mut(&'a mut self, range: RangeFrom<(usize, usize)>) -> Result<MutView<'a, T>> {
        let sz = self.size();
        self.slice_mut(range.start..sz)
    }
}

impl<'a, 'b, T> SliceMut<'a, RangeTo<(usize, usize)>> for MutView<'b, T> {
    type Slice = MutView<'a, T>;

    fn slice_mut(&'a mut self, range: RangeTo<(usize, usize)>) -> Result<MutView<'a, T>> {
        self.slice_mut((0, 0)..range.end)
    }
}

impl<'a, T> SliceMut<'a, Range<usize>> for RowVec<T> {
    type Slice = MutRow<'a, T>;

    fn slice_mut(&'a mut self, range: Range<usize>) -> Result<MutRow<'a, T>> {
        unsafe {
            mem::transmute(::Slice::slice(&*self.0, range))
        }
    }
}

impl<'a, T> SliceMut<'a, RangeFrom<usize>> for RowVec<T> {
    type Slice = MutRow<'a, T>;

    fn slice_mut(&'a mut self, range: RangeFrom<usize>) -> Result<MutRow<'a, T>> {
        let n = self.len();
        self.slice_mut(range.start..n)
    }
}

impl<'a, T> SliceMut<'a, RangeTo<usize>> for RowVec<T> {
    type Slice = MutRow<'a, T>;

    fn slice_mut(&'a mut self, range: RangeTo<usize>) -> Result<MutRow<'a, T>> {
        self.slice_mut(0..range.end)
    }
}
