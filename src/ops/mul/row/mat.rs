use std::num::{One, Zero};
use std::ops::{IndexAssign, Range, RangeFrom, RangeFull, RangeTo};

use blas::Gemv;

use ops::{Mat, Product, Scaled};
use strided::Row;
use traits::Transpose;

// NOTE Secondary
impl<'a, 'b, T> IndexAssign<RangeFull, Scaled<Product<&'a Row<T>, &'b Mat<T>>>> for Row<T> where
    T: 'a + 'b + Gemv + Zero,
{
    fn index_assign(&mut self, _: RangeFull, rhs: Scaled<Product<&'a Row<T>, &'b Mat<T>>>) {
        self.t().index_assign(.., rhs.t())
    }
}

// NOTE Secondary
impl<'a, 'b, T> IndexAssign<RangeFull, Product<&'a Row<T>, &'b Mat<T>>> for Row<T> where
    T: Gemv + One + Zero,
{
    fn index_assign(&mut self, _: RangeFull, rhs: Product<&'a Row<T>, &'b Mat<T>>) {
        self.index_assign(.., Scaled(T::one(), rhs))
    }
}

// NOTE Secondary
impl<'a, 'b, T> IndexAssign<Range<u32>, Product<&'a Row<T>, &'b Mat<T>>> for Row<T> where
    T: Gemv + One + Zero,
{
    fn index_assign(&mut self, r: Range<u32>, rhs: Product<&Row<T>, &Mat<T>>) {
        self[r.start..r.end].index_assign(.., rhs)
    }
}

// NOTE Secondary
impl<'a, 'b, T> IndexAssign<RangeFrom<u32>, Product<&'a Row<T>, &'b Mat<T>>> for Row<T> where
    T: Gemv + One + Zero,
{
    fn index_assign(&mut self, r: RangeFrom<u32>, rhs: Product<&Row<T>, &Mat<T>>) {
        self[r.start..].index_assign(.., rhs)
    }
}

// NOTE Secondary
impl<'a, 'b, T> IndexAssign<RangeTo<u32>, Product<&'a Row<T>, &'b Mat<T>>> for Row<T> where
    T: Gemv + One + Zero,
{
    fn index_assign(&mut self, r: RangeTo<u32>, rhs: Product<&Row<T>, &Mat<T>>) {
        self[..r.end].index_assign(.., rhs)
    }
}

// NB All these "forward impls" shouldn't be necessary, but there is a limitation in the index
// overloading code. See rust-lang/rust#26218
impl<'a, 'b, T> IndexAssign<RangeFull, Product<&'a Row<T>, &'b Mat<T>>> for ::Row<T> where
    T: Gemv + One + Zero,
{
    fn index_assign(&mut self, _: RangeFull, rhs: Product<&Row<T>, &Mat<T>>) {
        IndexAssign::index_assign(&mut **self, .., rhs)
    }
}

impl<'a, 'b, T> IndexAssign<Range<u32>, Product<&'a Row<T>, &'b Mat<T>>> for ::Row<T> where
    T: Gemv + One + Zero,
{
    fn index_assign(&mut self, r: Range<u32>, rhs: Product<&Row<T>, &Mat<T>>) {
        IndexAssign::index_assign(&mut **self, r, rhs)
    }
}

impl<'a, 'b, T> IndexAssign<RangeFrom<u32>, Product<&'a Row<T>, &'b Mat<T>>> for ::Row<T> where
    T: Gemv + One + Zero,
{
    fn index_assign(&mut self, r: RangeFrom<u32>, rhs: Product<&Row<T>, &Mat<T>>) {
        IndexAssign::index_assign(&mut **self, r, rhs)
    }
}

impl<'a, 'b, T> IndexAssign<RangeTo<u32>, Product<&'a Row<T>, &'b Mat<T>>> for ::Row<T> where
    T: Gemv + One + Zero,
{
    fn index_assign(&mut self, r: RangeTo<u32>, rhs: Product<&Row<T>, &Mat<T>>) {
        IndexAssign::index_assign(&mut **self, r, rhs)
    }
}
