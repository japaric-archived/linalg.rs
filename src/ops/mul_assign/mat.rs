use std::ops::{Mul, MulAssign};

use blas::Scal;
use extract::Extract;

use traits::Matrix;

// TODO Manually add Complex<f32>/Complex<f64> implementations
// NOTE Secondary
impl<T, O> MulAssign<T> for ::Mat<T, O> where T: Scal {
    fn mul_assign(&mut self, alpha: T) {
        super::slice(&alpha, self.as_mut())
    }
}

// NOTE Secondary
impl<'a, T, O> MulAssign<&'a ::Mat<T, O>> for ::Mat<T, O> where T: Clone + Mul<Output=T> {
    fn mul_assign(&mut self, rhs: &::Mat<T, O>) {
        unsafe {
            assert_eq!(self.size(), rhs.size());

            let mut src = rhs.as_ref().iter();

            for dst in self.as_mut().iter_mut() {
                *dst = dst.clone() * src.next().extract().clone();
            }
        }
    }
}
