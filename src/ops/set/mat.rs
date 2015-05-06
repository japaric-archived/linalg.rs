use blas::Copy;

use ops::{set, self};
use traits::{Matrix, MatrixCols, MatrixColsMut, MatrixRows, MatrixRowsMut, Set, SliceMut};
use {Col, ColMut, Mat, Row, RowMut, Transposed, SubMat, SubMatMut};

// NOTE Core
impl<'a, T> Set<T> for SubMatMut<'a, T> where T: Copy {
    fn set(&mut self, value: T) {
        let ref x = value;

        if let Some(y) = self.as_slice_mut() {
            return set::slice(x, y);
        }

        if self.nrows() < self.ncols() {
            for RowMut(Row(ref mut y)) in self.rows_mut() {
                set::strided(x, y)
            }
        } else {
            for ColMut(Col(ref mut y)) in self.cols_mut() {
                set::strided(x, y)
            }
        }
    }
}

// NOTE Core
impl<'a, 'b, T> Set<SubMat<'a, T>> for SubMatMut<'b, T> where T: Copy {
    fn set(&mut self, rhs: SubMat<T>) {
        unsafe {
            assert_eq!(self.size(), rhs.size());

            if let (Some(y), Some(x)) = (self.as_slice_mut(), rhs.as_slice()) {
                return ops::copy_slice(x, y);
            }

            if self.nrows() < self.ncols() {
                for (RowMut(Row(ref mut y)), Row(ref x)) in self.rows_mut().zip(rhs.rows()) {
                    ops::copy_strided(x, y)
                }
            } else {
                for (ColMut(Col(ref mut y)), Col(ref x)) in self.cols_mut().zip(rhs.cols()) {
                    ops::copy_strided(x, y)
                }
            }
        }
    }
}

// NOTE Core
impl<'a, 'b, T> Set<Transposed<SubMat<'a, T>>> for SubMatMut<'b, T> where T: Copy {
    fn set(&mut self, rhs: Transposed<SubMat<T>>) {
        unsafe {
            assert_eq!(self.size(), rhs.size());

            if self.nrows() < self.ncols() {
                for (RowMut(Row(ref mut y)), Row(ref x)) in self.rows_mut().zip(rhs.rows()) {
                    ops::copy_strided(x, y)
                }
            } else {
                for (ColMut(Col(ref mut y)), Col(ref x)) in self.cols_mut().zip(rhs.cols()) {
                    ops::copy_strided(x, y)
                }
            }
        }
    }
}

// NOTE Secondary
impl<'a, T> Set<T> for Transposed<SubMatMut<'a, T>> where T: Copy {
    fn set(&mut self, value: T) {
        self.0.set(value)
    }
}

// NOTE Secondary
impl<'a, 'b, T> Set<SubMat<'a, T>> for Transposed<SubMatMut<'b, T>> where T: Copy {
    fn set(&mut self, rhs: SubMat<T>) {
        self.0.set(Transposed(rhs))
    }
}

// NOTE Secondary
impl<'a, 'b, T> Set<Transposed<SubMat<'a, T>>> for Transposed<SubMatMut<'b, T>> where T: Copy {
    fn set(&mut self, rhs: Transposed<SubMat<T>>) {
        self.0.set(rhs.0)
    }
}

// NOTE Forward
impl<T> Set<T> for Transposed<Mat<T>> where T: Copy {
    fn set(&mut self, value: T) {
        self.slice_mut(..).set(value)
    }
}

// NOTE Forward
impl<T> Set<T> for Mat<T> where T: Copy {
    fn set(&mut self, value: T) {
        self.slice_mut(..).set(value)
    }
}
