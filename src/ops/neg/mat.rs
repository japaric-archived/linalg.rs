use std::num::One;
use std::ops::Neg;

use ops::Scaled;

impl<'a, T, O> Neg for &'a ::Mat<T, O> where T: Neg<Output=T> + One {
    type Output = Scaled<&'a ::Mat<T, O>>;

    fn neg(self) -> Self::Output {
        Scaled(T::one().neg(), self)
    }
}
