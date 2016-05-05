use std::ops::{Neg, SubAssign};

use blas::Axpy;
use onezero::One;

use ops;
use {DiagMut, Diag};

// Combinations:
//
// LHS: DiagMut
// RHS: &T, T
//
// -> 2 implementations

// Core implementations
impl<'a, 'b, T> SubAssign<&'a T> for DiagMut<'b, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        let DiagMut(Diag(ref mut y)) = *self;
        let x = rhs;
        let ref alpha = T::one().neg();

        ops::axpy_strided_scalar(alpha, x, y)
    }
}

// "Forwarding" implementations
impl<'a, T> SubAssign<T> for DiagMut<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: T) {
        *self -= &rhs
    }
}
