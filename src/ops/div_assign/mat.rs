use std::ops::{Div, DivAssign};
use std::num::One;

use blas::Scal;

impl<T, O> DivAssign<T> for ::Mat<T, O> where T: Div<Output=T> + One + Scal {
    fn div_assign(&mut self, rhs: T) {
        *self *= T::one() / rhs;
    }
}
