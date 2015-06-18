use std::ops::{Neg, Sub};

use ops::Sum;

impl<'a, T> Sub<T> for &'a ::Row<T> where T: Neg<Output=T> {
    type Output = Sum<&'a ::Row<T>, T>;

    fn sub(self, rhs: T) -> Self::Output {
        Sum(self, -rhs)
    }
}
