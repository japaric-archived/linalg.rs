use std::mem;

use error::OutOfBounds;
use traits::{At, AtMut};
use {Col, ColVec, Diag, Mat, MutCol, MutDiag, MutRow, MutView, Row, RowVec, View};

impl<T> ::At<usize> for [T] {
    type Output = T;

    fn at(&self, index: usize) -> Result<&T, OutOfBounds> {
        if index < self.len() {
            Ok(unsafe { mem::transmute(self.as_ptr().offset(index as isize)) })
        } else {
            Err(OutOfBounds)
        }
    }
}

impl<'a, T> At<usize> for Col<'a, T> {
    type Output = T;

    fn at(&self, index: usize) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<T> At<usize> for ColVec<T> {
    type Output = T;

    fn at(&self, index: usize) -> Result<&T, OutOfBounds> {
        ::At::at(&*self.0, index)
    }
}

impl<'a, T> At<usize> for Diag<'a, T> {
    type Output = T;

    fn at(&self, index: usize) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<T> At<(usize, usize)> for Mat<T> {
    type Output = T;

    fn at(&self, (row, col): (usize, usize)) -> Result<&T, OutOfBounds> {
        let (nrows, ncols) = (self.nrows, self.ncols);

        if row < nrows && col < ncols {
            Ok(unsafe { mem::transmute(self.data.as_ptr().offset((col * nrows + row) as isize)) })
        } else {
            Err(OutOfBounds)
        }
    }
}

impl<'a, T> At<usize> for MutCol<'a, T> {
    type Output = T;

    fn at(&self, index: usize) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<'a, T> At<usize> for MutDiag<'a, T> {
    type Output = T;

    fn at(&self, index: usize) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<'a, T> At<usize> for MutRow<'a, T> {
    type Output = T;

    fn at(&self, index: usize) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<'a, T> At<(usize, usize)> for MutView<'a, T> {
    type Output = T;

    fn at(&self, (row, col): (usize, usize)) -> Result<&T, OutOfBounds> {
        self.0.at((row, col))
    }
}

impl<'a, T> At<usize> for Row<'a, T> {
    type Output = T;

    fn at(&self, index: usize) -> Result<&T, OutOfBounds> {
        self.0.at(index)
    }
}

impl<T> At<usize> for RowVec<T> {
    type Output = T;

    fn at(&self, index: usize) -> Result<&T, OutOfBounds> {
        ::At::at(&*self.0, index)
    }
}

impl<'a, T> At<(usize, usize)> for View<'a, T> {
    type Output = T;

    fn at(&self, (row, col): (usize, usize)) -> Result<&T, OutOfBounds> {
        self.0.at((row, col))
    }
}

impl<T> AtMut<usize> for ColVec<T> {
    type Output = T;

    fn at_mut(&mut self, index: usize) -> Result<&mut T, OutOfBounds> {
        unsafe { mem::transmute(::At::at(&*self.0, index)) }
    }
}

impl<T> AtMut<(usize, usize)> for Mat<T> {
    type Output = T;

    fn at_mut(&mut self, (row, col): (usize, usize)) -> Result<&mut T, OutOfBounds> {
        unsafe { mem::transmute(self.at((row, col))) }
    }
}

impl<'a, T> AtMut<usize> for MutCol<'a, T> {
    type Output = T;

    fn at_mut(&mut self, index: usize) -> Result<&mut T, OutOfBounds> {
        self.0.at_mut(index)
    }
}

impl<'a, T> AtMut<usize> for MutDiag<'a, T> {
    type Output = T;

    fn at_mut(&mut self, index: usize) -> Result<&mut T, OutOfBounds> {
        self.0.at_mut(index)
    }
}

impl<'a, T> AtMut<usize> for MutRow<'a, T> {
    type Output = T;

    fn at_mut(&mut self, index: usize) -> Result<&mut T, OutOfBounds> {
        self.0.at_mut(index)
    }
}

impl<'a, T> AtMut<(usize, usize)> for MutView<'a, T> {
    type Output = T;

    fn at_mut(&mut self, (row, col): (usize, usize)) -> Result<&mut T, OutOfBounds> {
        unsafe { mem::transmute(self.0.at((row, col))) }
    }
}

impl<T> AtMut<usize> for RowVec<T> {
    type Output = T;

    fn at_mut(&mut self, index: usize) -> Result<&mut T, OutOfBounds> {
        unsafe { mem::transmute(::At::at(&*self.0, index)) }
    }
}
