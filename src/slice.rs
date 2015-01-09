use std::mem;
use std::raw::{self, Repr};

use {
    Col, ColVec, Diag, Error, Mat, MutCol, MutDiag, MutRow, MutView, Result, Row, RowVec, View,
    strided,
};
use traits::{Matrix, Slice, SliceMut};

impl<'a, T> ::Slice<'a, usize> for [T] {
    type Slice = strided::Slice<'a, T>;

    fn slice(&'a self, start: usize, end: usize) -> Result<strided::Slice<'a, T>> {
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

macro_rules! from_to {
    ($ty:ty) => {
        fn slice_from(&'a self, start: usize) -> Result<$ty> {
            Slice::slice(self, start, self.len())
        }

        fn slice_to(&'a self, end: usize) -> Result<$ty> {
            Slice::slice(self, 0, end)
        }
    }
}

impl<'a, T> Slice<'a, usize> for ColVec<T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, start: usize, end: usize) -> Result<Col<'a, T>> {
        ::Slice::slice(&*self.0, start, end).map(Col)
    }

    from_to!(Col<'a, T>);
}

impl<'a, 'b, T> Slice<'a, usize> for Col<'b, T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, start: usize, end: usize) -> Result<Col<'a, T>> {
        self.0.slice(start, end).map(Col)
    }

    from_to!(Col<'a, T>);
}

impl<'a, 'b, T> Slice<'a, usize> for Diag<'b, T> {
    type Slice = Diag<'a, T>;

    fn slice(&'a self, start: usize, end: usize) -> Result<Diag<'a, T>> {
        self.0.slice(start, end).map(Diag)
    }

    from_to!(Diag<'a, T>);
}

impl<'a, T> Slice<'a, (usize, usize)> for Mat<T> {
    type Slice = View<'a, T>;

    fn slice(
        &'a self,
        (start_row, start_col): (usize, usize),
        (end_row, end_col): (usize, usize),
    ) -> Result<View<'a, T>> {
        let (nrows, ncols) = (self.nrows, self.ncols);

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

    fn slice_from(&'a self, start: (usize, usize)) -> Result<View<'a, T>> {
        Slice::slice(self, start, (self.nrows, self.ncols))
    }

    fn slice_to(&'a self, end: (usize, usize)) -> Result<View<'a, T>> {
        Slice::slice(self, (0, 0), end)
    }
}

impl<'a, 'b, T> Slice<'a, usize> for MutCol<'b, T> {
    type Slice = Col<'a, T>;

    fn slice(&'a self, start: usize, end: usize) -> Result<Col<'a, T>> {
        self.0.slice(start, end).map(Col)
    }

    from_to!(Col<'a, T>);
}

impl<'a, 'b, T> Slice<'a, usize> for MutDiag<'b, T> {
    type Slice = Diag<'a, T>;

    fn slice(&'a self, start: usize, end: usize) -> Result<Diag<'a, T>> {
        self.0.slice(start, end).map(Diag)
    }

    from_to!(Diag<'a, T>);
}

impl<'a, 'b, T> Slice<'a, usize> for MutRow<'b, T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, start: usize, end: usize) -> Result<Row<'a, T>> {
        self.0.slice(start, end).map(Row)
    }

    from_to!(Row<'a, T>);
}

impl<'a, T> Slice<'a, usize> for RowVec<T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, start: usize, end: usize) -> Result<Row<'a, T>> {
        ::Slice::slice(&*self.0, start, end).map(Row)
    }

    from_to!(Row<'a, T>);
}

impl<'a, 'b, T> Slice<'a, usize> for Row<'b, T> {
    type Slice = Row<'a, T>;

    fn slice(&'a self, start: usize, end: usize) -> Result<Row<'a, T>> {
        self.0.slice(start, end).map(Row)
    }

    from_to!(Row<'a, T>);
}

macro_rules! from_to2 {
    ($ty:ty) => {
        fn slice_from(&'a self, start: (usize, usize)) -> Result<$ty> {
            Slice::slice(self, start, self.size())
        }

        fn slice_to(&'a self, end: (usize, usize)) -> Result<$ty> {
            Slice::slice(self, (0, 0), end)
        }
    }
}

impl<'a, 'b, T> Slice<'a, (usize, usize)> for MutView<'b, T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, start: (usize, usize), end: (usize, usize)) -> Result<View<'a, T>> {
        unsafe { mem::transmute(self.0.slice(start, end)) }
    }

    from_to2!(View<'a, T>);
}

impl<'a, 'b, T> Slice<'a, (usize, usize)> for View<'b, T> {
    type Slice = View<'a, T>;

    fn slice(&'a self, start: (usize, usize), end: (usize, usize)) -> Result<View<'a, T>> {
        unsafe { mem::transmute(self.0.slice(start, end)) }
    }

    from_to2!(View<'a, T>);
}

macro_rules! from_to_mut {
    ($ty:ty) => {
        fn slice_from_mut(&'a mut self, start: usize) -> Result<$ty> {
            let end = self.len();

            SliceMut::slice_mut(self, start, end)
        }

        fn slice_to_mut(&'a mut self, end: usize) -> Result<$ty> {
            SliceMut::slice_mut(self, 0, end)
        }
    }
}

impl<'a, T> SliceMut<'a, usize> for ColVec<T> {
    type Slice = MutCol<'a, T>;

    fn slice_mut(&'a mut self, start: usize, end: usize) -> Result<MutCol<'a, T>> {
        unsafe { mem::transmute(::Slice::slice(&*self.0, start, end)) }
    }

    from_to_mut!(MutCol<'a, T>);
}

impl<'a, T> SliceMut<'a, (usize, usize)> for Mat<T> {
    type Slice = MutView<'a, T>;

    fn slice_mut(
        &'a mut self,
        start: (usize, usize),
        end: (usize, usize),
    ) -> Result<MutView<'a, T>> {
        unsafe { mem::transmute(self.slice(start, end)) }
    }

    fn slice_from_mut(&'a mut self, start: (usize, usize)) -> Result<MutView<'a, T>> {
        let end = (self.nrows, self.ncols);

        SliceMut::slice_mut(self, start, end)
    }

    fn slice_to_mut(&'a mut self, end: (usize, usize)) -> Result<MutView<'a, T>> {
        SliceMut::slice_mut(self, (0, 0), end)
    }
}

impl<'a, 'b, T> SliceMut<'a, usize> for MutCol<'b, T> {
    type Slice = MutCol<'a, T>;

    fn slice_mut(&'a mut self, start: usize, end: usize) -> Result<MutCol<'a, T>> {
        self.0.slice_mut(start, end).map(MutCol)
    }

    from_to_mut!(MutCol<'a, T>);
}

impl<'a, 'b, T> SliceMut<'a, usize> for MutDiag<'b, T> {
    type Slice = MutDiag<'a, T>;

    fn slice_mut(&'a mut self, start: usize, end: usize) -> Result<MutDiag<'a, T>> {
        self.0.slice_mut(start, end).map(MutDiag)
    }

    from_to_mut!(MutDiag<'a, T>);
}

impl<'a, 'b, T> SliceMut<'a, usize> for MutRow<'b, T> {
    type Slice = MutRow<'a, T>;

    fn slice_mut(&'a mut self, start: usize, end: usize) -> Result<MutRow<'a, T>> {
        self.0.slice_mut(start, end).map(MutRow)
    }

    from_to_mut!(MutRow<'a, T>);
}

impl<'a, 'b, T> SliceMut<'a, (usize, usize)> for MutView<'b, T> {
    type Slice = MutView<'a, T>;

    fn slice_mut(
        &'a mut self,
        start: (usize, usize),
        end: (usize, usize),
    ) -> Result<MutView<'a, T>> {
        unsafe { mem::transmute(self.0.slice(start, end)) }
    }

    fn slice_from_mut(&'a mut self, start: (usize, usize)) -> Result<MutView<'a, T>> {
        let end = self.size();

        SliceMut::slice_mut(self, start, end)
    }

    fn slice_to_mut(&'a mut self, end: (usize, usize)) -> ::Result<MutView<'a, T>> {
        SliceMut::slice_mut(self, (0, 0), end)
    }
}

impl<'a, T> SliceMut<'a, usize> for RowVec<T> {
    type Slice = MutRow<'a, T>;

    fn slice_mut(&'a mut self, start: usize, end: usize) -> Result<MutRow<'a, T>> {
        unsafe { mem::transmute(::Slice::slice(&*self.0, start, end)) }
    }

    from_to_mut!(MutRow<'a, T>);
}
