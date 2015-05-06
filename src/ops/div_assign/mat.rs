use std::ops::Div;

use assign::{DivAssign, MulAssign};
use blas::Scal;
use onezero::One;

use traits::SliceMut;
use {Mat, Transposed, SubMatMut};

// NOTE Secondary
impl<'a, T, A> DivAssign<A> for SubMatMut<'a, T> where
    T: Scal<A>,
    A: Div<Output=A> + One,
{
    fn div_assign(&mut self, alpha: A) {
        self.mul_assign(A::one() / alpha)
    }
}

// NOTE Secondary
impl<'a, T, A> DivAssign<A> for Transposed<SubMatMut<'a, T>> where
    A: Div<Output=A> + One,
    T: Scal<A>,
{
    fn div_assign(&mut self, alpha: A) {
        self.0.div_assign(alpha)
    }
}

// NOTE Forward
impl<T, A> DivAssign<A> for Mat<T> where
    A: Div<Output=A> + One,
    T: Scal<A>,
{
    fn div_assign(&mut self, alpha: A) {
        self.slice_mut(..).div_assign(alpha)
    }
}

// NOTE Forward
impl<T, A> DivAssign<A> for Transposed<Mat<T>> where
    A: Div<Output=A> + One,
    T: Scal<A>,
{
    fn div_assign(&mut self, alpha: A) {
        self.slice_mut(..).div_assign(alpha)
    }
}
