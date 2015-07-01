use std::num::One;
use std::ops::{Add, IndexAssign, RangeFull};

use blas::Axpy;

use ops::{Scaled, Sum};
use order::Order;

// NOTE Secondary
impl<'a, 'b, T, O> Add<Scaled<&'b ::Mat<T, O>>> for &'a mut ::Mat<T, O> where
    T: 'a + 'b + Axpy,
    O: Order,
{
    type Output = &'a mut ::Mat<T, O>;

    fn add(self, rhs: Scaled<&'b ::Mat<T, O>>) -> Self::Output {
        *self += rhs;
        self
    }
}

// NOTE Secondary
impl<'a, 'b, T, O> Add<Scaled<&'b ::Mat<T, O>>> for &'a ::Mat<T, O> {
    type Output = Sum<&'a ::Mat<T, O>, Scaled<&'b ::Mat<T, O>>>;

    fn add(self, rhs: Scaled<&'b ::Mat<T, O>>) -> Self::Output {
        Sum(self, rhs)
    }
}

// NOTE Secondary
impl<'a, 'b, T, O> Add<&'b mut ::Mat<T, O>> for Scaled<&'a ::Mat<T, O>> where
    T: 'a + 'b + Axpy,
    O: Order,
{
    type Output = &'b mut ::Mat<T, O>;

    fn add(self, rhs: &'b mut ::Mat<T, O>) -> Self::Output {
        rhs + self
    }
}

// NOTE Secondary
impl<'a, T, O> Add<&'a mut ::Mat<T, O>> for &'a ::Mat<T, O> where T: 'a + Axpy + One, O: Order {
    type Output = &'a mut ::Mat<T, O>;

    fn add(self, rhs: &'a mut ::Mat<T, O>) -> &'a mut ::Mat<T, O> {
        Scaled(T::one(), self) + rhs
    }
}

impl<'a, 'b, T> IndexAssign<RangeFull, Sum<&'a ::Mat<T, ::order::Row>, Scaled<&'b ::Mat<T, ::order::Row>>>> for ::Mat<T, ::order::Row> where
    T: 'a + 'b + Axpy + Clone,
{
    fn index_assign(&mut self, _: RangeFull, rhs: Sum<&'a ::Mat<T, ::order::Row>, Scaled<&'b ::Mat<T, ::order::Row>>>) {
        let Sum(lhs, rhs) = rhs;
        self[..] = lhs;
        *self += rhs;
    }
}
