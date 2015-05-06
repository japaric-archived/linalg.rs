use assign::MulAssign;
use blas::Scal;

use ops;
use traits::{MatrixColsMut, MatrixRowsMut, SliceMut};
use {Col, ColMut, Mat, RowMut, Row, Transposed, SubMatMut};

// Core implementations
impl<'a, T, A> MulAssign<A> for SubMatMut<'a, T> where T: Scal<A> {
    fn mul_assign(&mut self, alpha: A) {
        unsafe {
            let ref alpha = alpha;

            if let Some(x) = self.as_slice_mut() {
                return ops::scal_slice(alpha, x);
            }

            if self.0.nrows < self.0.ncols {
                for RowMut(Row(ref mut x)) in self.rows_mut() {
                    ops::scal_strided(alpha, x);
                }
            } else {
                for ColMut(Col(ref mut x)) in self.cols_mut() {
                    ops::scal_strided(alpha, x);
                }
            }
        }
    }
}

// Secondary implementations
impl<'a, T, A> MulAssign<A> for Transposed<SubMatMut<'a, T>> where T: Scal<A> {
    fn mul_assign(&mut self, alpha: A) {
        self.0.mul_assign(alpha)
    }
}

// "Forwarding" implementations
impl<T, A> MulAssign<A> for Mat<T> where T: Scal<A> {
    fn mul_assign(&mut self, alpha: A) {
        self.slice_mut(..).mul_assign(alpha)
    }
}

// Secondary implementations
impl<T, A> MulAssign<A> for Transposed<Mat<T>> where T: Scal<A> {
    fn mul_assign(&mut self, alpha: A) {
        self.slice_mut(..).mul_assign(alpha)
    }
}

