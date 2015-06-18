#![doc(hidden)]

use std::num::One;
use std::ops::{Div, Mul, Neg};

use ops::{Product, Scaled};
use order::Order;
use traits::Matrix;

impl<T, L, R> Div<T> for Product<L, R> where
    L: Matrix<Elem=T>,
    R: Matrix<Elem=T>,
    T: Div<Output=T> + One,
{
    type Output = Scaled<Product<L, R>>;

    fn div(self, rhs: T) -> Scaled<Product<L, R>> {
        Scaled(T::one() / rhs, self)
    }
}

impl<T, M> Div<T> for Scaled<M> where M: Matrix<Elem=T>, T: Div<Output=T> {
    type Output = Scaled<M>;

    fn div(self, rhs: T) -> Scaled<M> {
        Scaled(self.0 / rhs, self.1)
    }
}

impl<'a, T, O> Mul<T> for &'a ::Mat<T, O> where O: Order {
    type Output = Scaled<&'a ::Mat<T, O>>;

    fn mul(self, rhs: T) -> Scaled<&'a ::Mat<T, O>> {
        Scaled(rhs, self)
    }
}

impl<'a, T, O> Mul<T> for &'a mut ::Mat<T, O> where O: Order {
    type Output = Scaled<&'a mut ::Mat<T, O>>;

    fn mul(self, rhs: T) -> Scaled<&'a mut ::Mat<T, O>> {
        Scaled(rhs, self)
    }
}

impl<T, L, R> Mul<T> for Product<L, R> where L: Matrix<Elem=T>, R: Matrix<Elem=T> {
    type Output = Scaled<Product<L, R>>;

    fn mul(self, rhs: T) -> Scaled<Product<L, R>> {
        Scaled(rhs, self)
    }
}

impl<T, M> Mul<T> for Scaled<M> where M: Matrix<Elem=T>, T: Mul<Output=T> {
    type Output = Scaled<M>;

    fn mul(self, rhs: T) -> Scaled<M> {
        Scaled(self.0 * rhs, self.1)
    }
}

impl<T, M> Neg for Scaled<M> where M: Matrix<Elem=T>, T: Neg<Output=T> {
    type Output = Scaled<M>;

    fn neg(self) -> Scaled<M> {
        Scaled(self.0.neg(), self.1)
    }
}
