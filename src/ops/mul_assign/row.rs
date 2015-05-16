use std::ops::MulAssign;

use blas::Scal;

use ops;
use traits::SliceMut;
use {Row, RowMut, RowVec};

impl<'a, T, A> MulAssign<A> for RowMut<'a, T> where T: Scal<A> {
    fn mul_assign(&mut self, alpha: A) {
        unsafe {
            let RowMut(Row(ref mut x)) = *self;
            let ref alpha = alpha;

            ops::scal_strided(alpha, x)
        }
    }
}

impl<T, A> MulAssign<A> for RowVec<T> where T: Scal<A> {
    fn mul_assign(&mut self, alpha: A) {
        self.slice_mut(..) *= alpha
    }
}
