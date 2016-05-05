use std::ops::{Div, DivAssign};

use blas::Scal;
use onezero::One;

use traits::SliceMut;
use {ColMut, ColVec};

// NOTE Secondary
impl<'a, T, A> DivAssign<A> for ColMut<'a, T> where
    A: Div<Output=A> + One,
    T: Scal<A>,
{
    fn div_assign(&mut self, alpha: A) {
        *self *= A::one() / alpha
    }
}

// NOTE Forward
impl<T, A> DivAssign<A> for ColVec<T> where
    A: Div<Output=A> + One,
    T: Scal<A>,
{
    fn div_assign(&mut self, alpha: A) {
        self.slice_mut(..) /= alpha
    }
}
