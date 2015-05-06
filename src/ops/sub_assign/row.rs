use std::ops::Neg;

use assign::SubAssign;
use blas::{Axpy, Gemm, Gemv};
use onezero::{One, Zero};

use Forward;
use traits::{Slice, SliceMut, Transpose};
use {Chain, Product, Row, RowMut, RowVec, Scaled, Transposed, SubMat};

// Combinations:
//
// LHS: RowMut, RowVec
// RHS: &T, T, Product<Row, Chain>, Product<Row, Transposed<SubMat>>, Product<Row, SubMat>, Row,
//      &RowMut, &RowVec, Scaled<Product<Row, Chain>>, Scaled<Product<Row, Transposed<SubMat>>>,
//      Scaled<Product<Row, SubMat>>, Scaled<Row>
//
// -> 24 implementations

// 9 impls
// Secondary implementations
//
// All these implementations defer their work to `col += mat * col`
//
// Remember that `(A + B)^t = A^t + B^t`, therefore:
//
//      `row <- row + row * mat === row^t <- row^t + mat^t * row^t`
impl<'a, 'b, T> SubAssign<&'a T> for RowMut<'b, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        self.slice_mut(..).t().sub_assign(rhs)
    }
}

impl<'a, 'b, T> SubAssign<Scaled<Row<'a, T>>> for RowMut<'b, T> where
    T: Axpy + Neg<Output=T>,
{
    fn sub_assign(&mut self, rhs: Scaled<Row<T>>) {
        self.slice_mut(..).t().sub_assign(rhs.t())
    }
}

impl<'a, 'b, 'c, T> SubAssign<Scaled<Product<Row<'a, T>, Chain<'b, T>>>> for RowMut<'c, T> where
    T: Gemm + Gemv + Neg<Output=T> + One + Zero,
{
    fn sub_assign(&mut self, rhs: Scaled<Product<Row<T>, Chain<T>>>) {
        self.slice_mut(..).t().sub_assign(rhs.t())
    }
}

impl<'a, 'b, 'c, T>
SubAssign<Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>>> for RowMut<'c, T> where
    T: Gemv + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Scaled<Product<Row<T>, Transposed<SubMat<T>>>>) {
        self.slice_mut(..).t().sub_assign(rhs.t())
    }
}

impl<'a, 'b, 'c, T> SubAssign<Scaled<Product<Row<'a, T>, SubMat<'b, T>>>> for RowMut<'c, T> where
    T: Gemv + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Scaled<Product<Row<T>, SubMat<T>>>) {
        self.slice_mut(..).t().sub_assign(rhs.t())
    }
}

impl<'a, 'b, T> SubAssign<Row<'a, T>> for RowMut<'b, T> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Row<T>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, 'c, T> SubAssign<Product<Row<'a, T>, Chain<'b, T>>> for RowMut<'c, T> where
    T: Gemm + Gemv + Neg<Output=T> + One + Zero,
{
    fn sub_assign(&mut self, rhs: Product<Row<T>, Chain<T>>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, 'c, T>
SubAssign<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>> for RowMut<'c, T> where
    T: Gemv + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Product<Row<T>, Transposed<SubMat<T>>>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, 'c, T> SubAssign<Product<Row<'a, T>, SubMat<'b, T>>> for RowMut<'c, T> where
    T: Gemv + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Product<Row<T>, SubMat<T>>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

macro_rules! forward {
    ($lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, 'c, T> SubAssign<$rhs> for $lhs where T: Neg<Output=T>, $(T: $bound),+ {
                fn sub_assign(&mut self, rhs: $rhs) {
                    self.slice_mut(..).sub_assign(rhs.slice(..))
                }
            }
         )+
    }
}

// 3 impls
forward!(RowMut<'a, T> {
    &'b RowMut<'c, T> { Axpy, One },
    &'b RowVec<T> { Axpy, One },
});

impl<'a, T> SubAssign<T> for RowMut<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.sub_assign(&rhs)
    }
}

// 12 impls
forward!(RowVec<T> {
    Product<Row<'a, T>, Chain<'b, T>> { Gemm, Gemv, One, Zero },
    Product<Row<'a, T>, Transposed<SubMat<'b, T>>> { Gemv, One },
    Product<Row<'a, T>, SubMat<'b, T>> { Gemv, One },
    Row<'a, T> { Axpy, One },
    &'a RowMut<'b, T> { Axpy, One },
    &'a RowVec<T> { Axpy, One },
    Scaled<Product<Row<'a, T>, Chain<'b, T>>> { Gemm, Gemv, One, Zero },
    Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>> { Gemv, One },
    Scaled<Product<Row<'a, T>, SubMat<'b, T>>> { Gemv, One },
    Scaled<Row<'a, T>> { Axpy },
});

impl<'a, T> SubAssign<&'a T> for RowVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        self.slice_mut(..).sub_assign(rhs)
    }
}

impl<T> SubAssign<T> for RowVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.slice_mut(..).sub_assign(&rhs)
    }
}
