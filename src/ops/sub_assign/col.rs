use std::ops::Neg;

use assign::SubAssign;
use blas::{Axpy, Gemm, Gemv, Transpose};
use onezero::{One, Zero};

use Forward;
use ops::{Reduce, self};
use traits::{Matrix, Slice, SliceMut};
use {Chain, Col, ColMut, ColVec, Product, Scaled, Transposed, SubMat};

// Combinations:
//
// LHS: ColMut, ColVec
// RHS: &T, T, Col, &ColMut, &ColVec, Product<Chain, Col>, Product<Transposed<SubMat>, Col>,
//      Product<SubMat, Col>, Scaled<Col>, Scaled<Product<Chain, Col>>,
//      Scaled<Product<Transposed<SubMat>, Col>>, Scaled<Product<SubMat, Col>>
//
// -> 24 implementations

// 5 impls
// Core implementations
impl<'a, 'b, T> SubAssign<&'a T> for ColMut<'b, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        let ref alpha = T::one().neg();
        let x = rhs;
        let ColMut(Col(ref mut y)) = *self;

        ops::axpy_strided_scalar(alpha, x, y)
    }
}

impl<'a, 'b, 'c, T> SubAssign<Scaled<Product<Chain<'a, T>, Col<'b, T>>>> for ColMut<'c, T> where
    T: Gemm + Gemv + Neg<Output=T> + One + Zero,
{
    fn sub_assign(&mut self, rhs: Scaled<Product<Chain<T>, Col<T>>>) {
        unsafe {
            use ops::reduce::MatMulCol::*;

            assert_eq_size!(self, rhs);

            let Scaled(alpha, rhs) = rhs;
            let ref alpha = alpha.neg();

            let a_mul_b = rhs.reduce();

            let (ref transa, a, x) = match a_mul_b {
                M_C(ref lhs, x) => {
                    (Transpose::No, lhs.slice(..), x)
                },
                M_CV(ref lhs, ref rhs) => {
                    (Transpose::No, lhs.slice(..), rhs.slice(..))
                },
                SM_CV((transa, a), ref rhs) => {
                    (transa, a, rhs.slice(..))
                }
            };

            let y = self.slice_mut(..);
            let ref beta = T::one();

            ops::gemv(transa, alpha, a, beta, x, y)
        }
    }
}

impl<'a, 'b, 'c, T>
SubAssign<Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>> for ColMut<'c, T> where
    T: Gemv + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Scaled<Product<Transposed<SubMat<T>>, Col<T>>>) {
        unsafe {
            assert_eq_size!(self, rhs);

            let ref trans = Transpose::Yes;
            let Scaled(alpha, Product(Transposed(a), x)) = rhs;
            let ref alpha = alpha.neg();
            let ref beta = T::one();
            let y = ColMut(self.0);

            ops::gemv(trans, alpha, a, beta, x, y);
        }
    }
}

impl<'a, 'b, 'c, T> SubAssign<Scaled<Product<SubMat<'a, T>, Col<'b, T>>>> for ColMut<'c, T> where
    T: Gemv + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Scaled<Product<SubMat<T>, Col<T>>>) {
        unsafe {
            assert_eq_size!(self, rhs);

            let ref trans = Transpose::No;
            let Scaled(alpha, Product(a, x)) = rhs;
            let ref alpha = alpha.neg();
            let ref beta = T::one();
            let y = ColMut(self.0);

            ops::gemv(trans, alpha, a, beta, x, y);
        }
    }
}

impl<'a, 'b, T> SubAssign<Scaled<Col<'a, T>>> for ColMut<'b, T> where T: Axpy + Neg<Output=T> {
    fn sub_assign(&mut self, rhs: Scaled<Col<T>>) {
        unsafe {
            assert_eq_size!(self, rhs);

            let ColMut(Col(ref mut y)) = *self;
            let Scaled(alpha, col) = rhs;
            let ref alpha = alpha.neg();
            let Col(ref x) = col;

            ops::axpy_strided_strided(alpha, x, y);
        }
    }
}

// 4 impls
// Secondary implementations
impl<'a, 'b, T> SubAssign<Col<'a, T>> for ColMut<'b, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: Col<T>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, 'c, T> SubAssign<Product<Chain<'a, T>, Col<'b, T>>> for ColMut<'c, T> where
    T: Gemm + Gemv + Neg<Output=T> + One + Zero,
{
    fn sub_assign(&mut self, rhs: Product<Chain<T>, Col<T>>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, 'c, T>
SubAssign<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> for ColMut<'c, T> where
    T: Gemv + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Product<Transposed<SubMat<T>>, Col<T>>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, 'c, T> SubAssign<Product<SubMat<'a, T>, Col<'b, T>>> for ColMut<'c, T> where
    T: Gemv + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Product<SubMat<T>, Col<T>>) {
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
forward!(ColMut<'a, T> {
    &'b ColMut<'c, T> { Axpy, One },
    &'b ColVec<T> { Axpy, One },
});

impl<'a, T> SubAssign<T> for ColMut<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.sub_assign(&rhs)
    }
}

// 12 impls
forward!(ColVec<T> {
    Col<'a, T> { Axpy, One },
    &'a ColMut<'b, T> { Axpy, One },
    &'a ColVec<T> { Axpy, One },
    Product<Chain<'a, T>, Col<'b, T>> { Gemm, Gemv, One, Zero },
    Product<Transposed<SubMat<'a, T>>, Col<'b, T>> { Gemv, One },
    Product<SubMat<'a, T>, Col<'b, T>> { Gemv, One },
    Scaled<Col<'a, T>> { Axpy },
    Scaled<Product<Chain<'a, T>, Col<'b, T>>> { Gemm, Gemv, One, Zero },
    Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> { Gemv, One },
    Scaled<Product<SubMat<'a, T>, Col<'b, T>>> { Gemv, One },
});

impl<'a, T> SubAssign<&'a T> for ColVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        self.slice_mut(..).sub_assign(rhs)
    }
}

impl<T> SubAssign<T> for ColVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.slice_mut(..).sub_assign(&rhs)
    }
}
