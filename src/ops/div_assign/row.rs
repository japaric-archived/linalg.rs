use std::ops::Div;

use assign::{DivAssign, MulAssign};
use blas::Scal;
use onezero::One;

use traits::SliceMut;
use {RowMut, RowVec};

// NOTE Secondary
impl<'a, T, A> DivAssign<A> for RowMut<'a, T> where
    A: Div<Output=A> + One,
    T: Scal<A>,
{
    fn div_assign(&mut self, alpha: A) {
        self.mul_assign(A::one() / alpha)
    }
}

// NOTE Secondary
impl<T, A> DivAssign<A> for RowVec<T> where
    A: Div<Output=A> + One,
    T: Scal<A>,
{
    fn div_assign(&mut self, alpha: A) {
        self.slice_mut(..).div_assign(alpha)
    }
}
