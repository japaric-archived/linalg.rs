#![doc(hidden)]

use std::ops::Neg;

use ops::Sum;

mod col;

impl<L, R> Neg for Sum<L, R> where L: Neg, R: Neg {
    type Output = Sum<L::Output, R::Output>;

    fn neg(self) -> Sum<L::Output, R::Output> {
        Sum(self.0.neg(), self.1.neg())
    }
}
