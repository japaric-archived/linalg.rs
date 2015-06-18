use std::ops::MulAssign;

use blas::Scal;

// NOTE Secondary
impl<T, A> MulAssign<A> for ::Row<T> where T: Scal<A> {
    fn mul_assign(&mut self, alpha: A) {
        super::slice(&alpha, self.as_mut())
    }
}
