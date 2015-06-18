use std::num::{One, Zero};
use std::ops::{IndexAssign, Range, RangeFrom, RangeFull, RangeTo};

use blas::Gemm;

use ops::{Mat, Product, Scaled};
use order::Order;

// NOTE Core
impl<'a, 'b, T> IndexAssign<RangeFull, Scaled<Product<&'a Mat<T>, &'b Mat<T>>>> for Mat<T> where
    T: 'a + 'b + Gemm + Zero,
{
    fn index_assign(&mut self, _: RangeFull, rhs: Scaled<Product<&'a Mat<T>, &'b Mat<T>>>) {
        let Scaled(ref alpha, Product(a, b)) = rhs;
        let c = self;
        let ref beta = T::zero();

        ::ops::blas::gemm(alpha, a, b, beta, c)
    }
}

// NOTE Secondary
impl<'a, 'b, T> IndexAssign<RangeFull, Product<&'a Mat<T>, &'b Mat<T>>> for Mat<T> where
    T: Gemm + One + Zero,
{
    fn index_assign(&mut self, _: RangeFull, rhs: Product<&'a Mat<T>, &'b Mat<T>>) {
        self.index_assign(.., Scaled(T::one(), rhs))
    }
}

// NOTE Secondary
impl<'a, 'b, T> IndexAssign<(RangeFull, Range<u32>), Product<&'a Mat<T>, &'b Mat<T>>> for ::Mat<T, ::order::Col> where
    T: Gemm + One + Zero,
{
    fn index_assign(&mut self, (r, c): (RangeFull, Range<u32>), rhs: Product<&'a Mat<T>, &'b Mat<T>>) {
        ::Mat::index_assign(&mut self[r, c], .., rhs)
    }
}

// NOTE Secondary
impl<'a, 'b, T> IndexAssign<(RangeFull, RangeFrom<u32>), Product<&'a Mat<T>, &'b Mat<T>>> for ::Mat<T, ::order::Col> where
    T: Gemm + One + Zero,
{
    fn index_assign(&mut self, (r, c): (RangeFull, RangeFrom<u32>), rhs: Product<&'a Mat<T>, &'b Mat<T>>) {
        ::Mat::index_assign(&mut self[r, c], .., rhs)
    }
}

// NOTE Secondary
impl<'a, 'b, T> IndexAssign<(RangeFull, RangeTo<u32>), Product<&'a Mat<T>, &'b Mat<T>>> for ::Mat<T, ::order::Col> where
    T: Gemm + One + Zero,
{
    fn index_assign(&mut self, (r, c): (RangeFull, RangeTo<u32>), rhs: Product<&'a Mat<T>, &'b Mat<T>>) {
        ::Mat::index_assign(&mut self[r, c], .., rhs)
    }
}

// NB All these "forward impls" shouldn't be necessary, but there is a limitation in the index
// overloading code. See rust-lang/rust#26218
impl<'a, 'b, T, O> IndexAssign<RangeFull, Product<&'a Mat<T>, &'b Mat<T>>> for ::Mat<T, O> where
    O: Order,
    T: Gemm + One + Zero,
{
    fn index_assign(&mut self, _: RangeFull, rhs: Product<&'a Mat<T>, &'b Mat<T>>) {
        Mat::index_assign(&mut ***self, .., rhs)
    }
}
