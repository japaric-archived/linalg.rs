use std::mem;
use std::raw;
use std::raw::Repr;

use {
    Col, ColVec, Diag, Error, Mat, MutCol, MutDiag, MutRow, MutView, Result, Row, RowVec, View,
    strided,
};
use traits::{Matrix, Slice, SliceMut};

impl<'a, T> ::Slice<'a, uint, strided::Slice<'a, T>> for [T] {
    fn slice(&'a self, start: uint, end: uint) -> Result<strided::Slice<'a, T>> {
        let raw::Slice { data, len } = self.repr();

        if start > end {
            Err(Error::InvalidSlice)
        } else if end > len {
            Err(Error::OutOfBounds)
        } else {
            Ok(unsafe { ::From::parts((
                data.offset(start as int),
                end - start,
                1,
            ))})
        }
    }
}

macro_rules! from_to {
    ($ty:ty) => {
        fn slice_from(&'a self, start: uint) -> Result<$ty> {
            Slice::slice(self, start, self.len())
        }

        fn slice_to(&'a self, end: uint) -> Result<$ty> {
            Slice::slice(self, 0, end)
        }
    }
}

impl<'a, T> Slice<'a, uint, Col<'a, T>> for ColVec<T> {
    fn slice(&'a self, start: uint, end: uint) -> Result<Col<'a, T>> {
        ::Slice::slice(&*self.0, start, end).map(Col)
    }

    from_to!(Col<'a, T>);
}

impl<'a, 'b, T> Slice<'a, uint, Col<'a, T>> for Col<'b, T> {
    fn slice(&'a self, start: uint, end: uint) -> Result<Col<'a, T>> {
        self.0.slice(start, end).map(Col)
    }

    from_to!(Col<'a, T>);
}

impl<'a, 'b, T> Slice<'a, uint, Diag<'a, T>> for Diag<'b, T> {
    fn slice(&'a self, start: uint, end: uint) -> Result<Diag<'a, T>> {
        self.0.slice(start, end).map(Diag)
    }

    from_to!(Diag<'a, T>);
}

impl<'a, T> Slice<'a, (uint, uint), View<'a, T>> for Mat<T> {
    fn slice(
        &'a self,
        (start_row, start_col): (uint, uint),
        (end_row, end_col): (uint, uint),
    ) -> Result<View<'a, T>> {
        let (nrows, ncols) = (self.nrows, self.ncols);

        if end_col > ncols || end_row > nrows {
            Err(Error::OutOfBounds)
        } else if start_col > end_col || start_row > end_row {
            Err(Error::InvalidSlice)
        } else {
            let stride = nrows;
            let ptr = unsafe {
                self.data.as_ptr().offset((start_col * stride + start_row) as int)
            };

            Ok(unsafe { ::From::parts((
                ptr,
                end_row - start_row,
                end_col - start_col,
                stride,
            ))})
        }
    }

    fn slice_from(&'a self, start: (uint, uint)) -> Result<View<'a, T>> {
        Slice::slice(self, start, (self.nrows, self.ncols))
    }

    fn slice_to(&'a self, end: (uint, uint)) -> Result<View<'a, T>> {
        Slice::slice(self, (0, 0), end)
    }
}

impl<'a, 'b, T> Slice<'a, uint, Col<'a, T>> for MutCol<'b, T> {
    fn slice(&'a self, start: uint, end: uint) -> Result<Col<'a, T>> {
        self.0.slice(start, end).map(Col)
    }

    from_to!(Col<'a, T>);
}

impl<'a, 'b, T> Slice<'a, uint, Diag<'a, T>> for MutDiag<'b, T> {
    fn slice(&'a self, start: uint, end: uint) -> Result<Diag<'a, T>> {
        self.0.slice(start, end).map(Diag)
    }

    from_to!(Diag<'a, T>);
}

impl<'a, 'b, T> Slice<'a, uint, Row<'a, T>> for MutRow<'b, T> {
    fn slice(&'a self, start: uint, end: uint) -> Result<Row<'a, T>> {
        self.0.slice(start, end).map(Row)
    }

    from_to!(Row<'a, T>);
}

impl<'a, T> Slice<'a, uint, Row<'a, T>> for RowVec<T> {
    fn slice(&'a self, start: uint, end: uint) -> Result<Row<'a, T>> {
        ::Slice::slice(&*self.0, start, end).map(Row)
    }

    from_to!(Row<'a, T>);
}

impl<'a, 'b, T> Slice<'a, uint, Row<'a, T>> for Row<'b, T> {
    fn slice(&'a self, start: uint, end: uint) -> Result<Row<'a, T>> {
        self.0.slice(start, end).map(Row)
    }

    from_to!(Row<'a, T>);
}

macro_rules! from_to2 {
    ($ty:ty) => {
        fn slice_from(&'a self, start: (uint, uint)) -> Result<$ty> {
            Slice::slice(self, start, self.size())
        }

        fn slice_to(&'a self, end: (uint, uint)) -> Result<$ty> {
            Slice::slice(self, (0, 0), end)
        }
    }
}

impl<'a, 'b, T> Slice<'a, (uint, uint), View<'a, T>> for MutView<'b, T> {
    fn slice(&'a self, start: (uint, uint), end: (uint, uint)) -> Result<View<'a, T>> {
        unsafe { mem::transmute(self.0.slice(start, end)) }
    }

    from_to2!(View<'a, T>);
}

impl<'a, 'b, T> Slice<'a, (uint, uint), View<'a, T>> for View<'b, T> {
    fn slice(&'a self, start: (uint, uint), end: (uint, uint)) -> Result<View<'a, T>> {
        unsafe { mem::transmute(self.0.slice(start, end)) }
    }

    from_to2!(View<'a, T>);
}

macro_rules! from_to_mut {
    ($ty:ty) => {
        fn slice_from_mut(&'a mut self, start: uint) -> Result<$ty> {
            let end = self.len();

            SliceMut::slice_mut(self, start, end)
        }

        fn slice_to_mut(&'a mut self, end: uint) -> Result<$ty> {
            SliceMut::slice_mut(self, 0, end)
        }
    }
}

impl<'a, T> SliceMut<'a, uint, MutCol<'a, T>> for ColVec<T> {
    fn slice_mut(&'a mut self, start: uint, end: uint) -> Result<MutCol<'a, T>> {
        unsafe { mem::transmute(::Slice::slice(&*self.0, start, end)) }
    }

    from_to_mut!(MutCol<'a, T>);
}

impl<'a, T> SliceMut<'a, (uint, uint), MutView<'a, T>> for Mat<T> {
    fn slice_mut(&'a mut self, start: (uint, uint), end: (uint, uint)) -> Result<MutView<'a, T>> {
        unsafe { mem::transmute(self.slice(start, end)) }
    }

    fn slice_from_mut(&'a mut self, start: (uint, uint)) -> Result<MutView<'a, T>> {
        let end = (self.nrows, self.ncols);

        SliceMut::slice_mut(self, start, end)
    }

    fn slice_to_mut(&'a mut self, end: (uint, uint)) -> Result<MutView<'a, T>> {
        SliceMut::slice_mut(self, (0, 0), end)
    }
}

impl<'a, 'b, T> SliceMut<'a, uint, MutCol<'a, T>> for MutCol<'b, T> {
    fn slice_mut(&'a mut self, start: uint, end: uint) -> Result<MutCol<'a, T>> {
        self.0.slice_mut(start, end).map(MutCol)
    }

    from_to_mut!(MutCol<'a, T>);
}

impl<'a, 'b, T> SliceMut<'a, uint, MutDiag<'a, T>> for MutDiag<'b, T> {
    fn slice_mut(&'a mut self, start: uint, end: uint) -> Result<MutDiag<'a, T>> {
        self.0.slice_mut(start, end).map(MutDiag)
    }

    from_to_mut!(MutDiag<'a, T>);
}

impl<'a, 'b, T> SliceMut<'a, uint, MutRow<'a, T>> for MutRow<'b, T> {
    fn slice_mut(&'a mut self, start: uint, end: uint) -> Result<MutRow<'a, T>> {
        self.0.slice_mut(start, end).map(MutRow)
    }

    from_to_mut!(MutRow<'a, T>);
}

impl<'a, 'b, T> SliceMut<'a, (uint, uint), MutView<'a, T>> for MutView<'b, T> {
    fn slice_mut(&'a mut self, start: (uint, uint), end: (uint, uint)) -> Result<MutView<'a, T>> {
        unsafe { mem::transmute(self.0.slice(start, end)) }
    }

    fn slice_from_mut(&'a mut self, start: (uint, uint)) -> Result<MutView<'a, T>> {
        let end = self.size();

        SliceMut::slice_mut(self, start, end)
    }

    fn slice_to_mut(&'a mut self, end: (uint, uint)) -> ::Result<MutView<'a, T>> {
        SliceMut::slice_mut(self, (0, 0), end)
    }
}

impl<'a, T> SliceMut<'a, uint, MutRow<'a, T>> for RowVec<T> {
    fn slice_mut(&'a mut self, start: uint, end: uint) -> Result<MutRow<'a, T>> {
        unsafe { mem::transmute(::Slice::slice(&*self.0, start, end)) }
    }

    from_to_mut!(MutRow<'a, T>);
}
