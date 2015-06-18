use std::num::One;
use std::ops::AddAssign;

use blas::Axpy;

use ops::Scaled;

// NOTE Core
impl<'a, T> AddAssign<Scaled<&'a ::Row<T>>> for ::Row<T> where T: 'a + Axpy {
    fn add_assign(&mut self, rhs: Scaled<&'a ::Row<T>>) {
        let Scaled(ref alpha, x) = rhs;
        let y = self;

        ::ops::blas::axpy(alpha, x, y);
    }
}

// NOTE Core
impl<'a, T> AddAssign<&'a ::Row<T>> for ::Row<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &'a ::Row<T>) {
        *self += Scaled(T::one(), rhs)
    }
}
