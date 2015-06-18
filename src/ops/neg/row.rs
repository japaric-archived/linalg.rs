use std::num::One;
use std::ops::Neg;

use ops::Scaled;

impl<'a, T> Neg for &'a ::Row<T> where T: Neg<Output=T> + One {
    type Output = Scaled<&'a ::Row<T>>;

    fn neg(self) -> Self::Output {
        Scaled(T::one().neg(), self)
    }
}
