use std::num::One;
use std::ops::{Neg, Sub};

use blas::Axpy;

use ops::{Scaled, Sum};
use order::Order;

impl<'a, 'b, T, O> Sub<Scaled<&'b ::Mat<T, O>>> for &'a mut ::Mat<T, O> where
    T: 'a + 'b + Axpy + Neg<Output=T>,
    O: Order,
{
    type Output = &'a mut ::Mat<T, O>;

    fn sub(self, rhs: Scaled<&'b ::Mat<T, O>>) -> &'a mut ::Mat<T, O> {
        self + rhs.neg()
    }
}

impl<'a, 'b, T, O> Sub<&'b ::Mat<T, O>> for &'a ::Mat<T, O> where
    T: 'a + 'b + Neg<Output=T> + One,
    O: Order,
{
    type Output = Sum<&'a ::Mat<T, O>, Scaled<&'b ::Mat<T, O>>>;

    fn sub(self, rhs: &'b ::Mat<T, O>) -> Self::Output {
        self + rhs.neg()
    }
}
