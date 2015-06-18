use std::ops::Mul;

use blas::Dot;

// NOTE Core
impl<'a, 'b, T> Mul<&'b ::strided::Col<T>> for &'a ::strided::Row<T> where T: Dot {
    type Output = T;

    fn mul(self, rhs: &'b ::strided::Col<T>) -> T {
        ::ops::blas::dot(self, rhs)
    }
}

// NOTE Forward
impl<'a, 'b, T> Mul<&'b ::Col<T>> for &'a ::Row<T> where T: Dot {
    type Output = T;

    fn mul(self, rhs: &'b ::Col<T>) -> T {
        Mul::mul(&**self, &**rhs)
    }
}
