use assign::MulAssign;
use blas::Scal;

use ops;
use traits::SliceMut;
use {Col, ColMut, ColVec};

impl<'a, T, A> MulAssign<A> for ColMut<'a, T> where T: Scal<A> {
    fn mul_assign(&mut self, alpha: A) {
        unsafe {
            let ColMut(Col(ref mut x)) = *self;
            let ref alpha = alpha;

            ops::scal_strided(alpha, x)
        }
    }
}

impl<T, A> MulAssign<A> for ColVec<T> where T: Scal<A> {
    fn mul_assign(&mut self, alpha: A) {
        self.slice_mut(..).mul_assign(alpha)
    }
}
