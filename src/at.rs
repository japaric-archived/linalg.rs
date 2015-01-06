use std::mem;

use error::OutOfBounds;
use traits::{At, AtMut};
use {Col, ColVec, Diag, Mat, MutCol, MutDiag, MutRow, MutView, Row, RowVec, View};

impl<T> ::At<uint> for [T] {
    type Output = T;

    fn at(&self, index: uint) -> Result<&T, OutOfBounds> {
        if index < self.len() {
            Ok(unsafe { mem::transmute(self.as_ptr().offset(index as int)) })
        } else {
            Err(OutOfBounds)
        }
    }
}

impl<'a, T> At<uint> for Col<'a, T> {
    type Output = T;

    fn at(&self, index: uint) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<T> At<uint> for ColVec<T> {
    type Output = T;

    fn at(&self, index: uint) -> Result<&T, OutOfBounds> {
        ::At::at(&*self.0, index)
    }
}

impl<'a, T> At<uint> for Diag<'a, T> {
    type Output = T;

    fn at(&self, index: uint) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<T> At<(uint, uint)> for Mat<T> {
    type Output = T;

    fn at(&self, (row, col): (uint, uint)) -> Result<&T, OutOfBounds> {
        let (nrows, ncols) = (self.nrows, self.ncols);

        if row < nrows && col < ncols {
            Ok(unsafe { mem::transmute(self.data.as_ptr().offset((col * nrows + row) as int)) })
        } else {
            Err(OutOfBounds)
        }
    }
}

impl<'a, T> At<uint> for MutCol<'a, T> {
    type Output = T;

    fn at(&self, index: uint) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<'a, T> At<uint> for MutDiag<'a, T> {
    type Output = T;

    fn at(&self, index: uint) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<'a, T> At<uint> for MutRow<'a, T> {
    type Output = T;

    fn at(&self, index: uint) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<'a, T> At<(uint, uint)> for MutView<'a, T> {
    type Output = T;

    fn at(&self, (row, col): (uint, uint)) -> Result<&T, OutOfBounds> {
        self.0.at((row, col))
    }
}

impl<'a, T> At<uint> for Row<'a, T> {
    type Output = T;

    fn at(&self, index: uint) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<T> At<uint> for RowVec<T> {
    type Output = T;

    fn at(&self, index: uint) -> Result<&T, OutOfBounds> {
        ::At::at(&*self.0, index)
    }
}

impl<'a, T> At<(uint, uint)> for View<'a, T> {
    type Output = T;

    fn at(&self, (row, col): (uint, uint)) -> Result<&T, OutOfBounds> {
        self.0.at((row, col))
    }
}

impl<T> AtMut<uint> for ColVec<T> {
    type Output = T;

    fn at_mut(&mut self, index: uint) -> Result<&mut T, OutOfBounds> {
        unsafe { mem::transmute(::At::at(&*self.0, index)) }
    }
}

impl<T> AtMut<(uint, uint)> for Mat<T> {
    type Output = T;

    fn at_mut(&mut self, (row, col): (uint, uint)) -> Result<&mut T, OutOfBounds> {
        unsafe { mem::transmute(self.at((row, col))) }
    }
}

impl<'a, T> AtMut<uint> for MutCol<'a, T> {
    type Output = T;

    fn at_mut(&mut self, index: uint) -> Result<&mut T, OutOfBounds> {
        self.0.at_mut(index)
    }
}

impl<'a, T> AtMut<uint> for MutDiag<'a, T> {
    type Output = T;

    fn at_mut(&mut self, index: uint) -> Result<&mut T, OutOfBounds> {
        self.0.at_mut(index)
    }
}

impl<'a, T> AtMut<uint> for MutRow<'a, T> {
    type Output = T;

    fn at_mut(&mut self, index: uint) -> Result<&mut T, OutOfBounds> {
        self.0.at_mut(index)
    }
}

impl<'a, T> AtMut<(uint, uint)> for MutView<'a, T> {
    type Output = T;

    fn at_mut(&mut self, (row, col): (uint, uint)) -> Result<&mut T, OutOfBounds> {
        unsafe { mem::transmute(self.0.at((row, col))) }
    }
}

impl<T> AtMut<uint> for RowVec<T> {
    type Output = T;

    fn at_mut(&mut self, index: uint) -> Result<&mut T, OutOfBounds> {
        unsafe { mem::transmute(::At::at(&*self.0, index)) }
    }
}
