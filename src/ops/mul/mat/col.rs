use std::ops::{IndexAssign, RangeFull};
use std::num::Zero;

use blas::Gemv;

use ops::{Mat, Product, Scaled};
use strided::Col;

// NOTE Core
impl<'a, 'b, T> IndexAssign<RangeFull, Scaled<Product<&'a Mat<T>, &'b Col<T>>>> for Col<T> where
    T: 'a + 'b + Gemv + Zero,
{
    fn index_assign(&mut self, _: RangeFull, rhs: Scaled<Product<&'a Mat<T>, &'b Col<T>>>) {
        let Scaled(ref alpha, Product(a, x)) = rhs;
        let y = self;

        ::ops::blas::gemv(alpha, a, x, &T::zero(), y);
    }
}
