use std::num::One;
use std::ops::{Neg, SubAssign};

use blas::Axpy;

use ops::Scaled;
use order::Order;

// NOTE Secondary
impl<'a, T, O> SubAssign<Scaled<&'a ::Mat<T, O>>> for ::Mat<T, O> where
    T: 'a + Axpy + Neg<Output=T>,
    O: Order,
{
    fn sub_assign(&mut self, rhs: Scaled<&'a ::Mat<T, O>>) {
        *self += -rhs;
    }
}

// NOTE Secondary
impl<'a, T, O> SubAssign<&'a ::Mat<T, O>> for ::Mat<T, O> where
    T: 'a + Axpy + Neg<Output=T> + One,
    O: Order,
{
    fn sub_assign(&mut self, rhs: &'a ::Mat<T, O>) {
        *self += Scaled(T::one().neg(), rhs);
    }
}
