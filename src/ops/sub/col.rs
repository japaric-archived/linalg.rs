use std::ops::{Neg, Sub};

use blas::{Axpy, Gemm, Gemv, Transpose};
use onezero::{One, Zero};

use ops::{Reduce, self};
use traits::{Matrix, Slice, SliceMut};
use {Chain, Col, ColMut, ColVec, Product, Scaled, Transposed, SubMat};

// GEMV
// Combinations:
//
// LHS: ColVec, Scaled<ColVec>
// RHS: Product<Chain, Col>, Product<Transposed<SubMat>, Col>, Product<SubMat, Col>,
//      Scaled<Product<Chain, Col>>, Scaled<Product<Transposed<SubMat>, Col>>
//      Scaled<Product<SubMat, Col>>
//
// and the reverse operation
//
// -> 24 implementations

// 6 impls
// Core implementations
impl<'a, 'b, T> Sub<Scaled<Product<Chain<'a, T>, Col<'b, T>>>> for Scaled<ColVec<T>> where
    T: Gemm + Gemv + Neg<Output=T> + One + Zero,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Scaled<Product<Chain<T>, Col<T>>>) -> ColVec<T> {
        unsafe {
            use ops::reduce::MatMulCol::*;

            assert_eq_size!(self, rhs);

            let Scaled(alpha, mut y) = self;
            let ref alpha = alpha.neg();
            let Scaled(beta, rhs) = rhs;
            let ref beta = beta;

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

            ops::gemv(transa, alpha, a, beta, x, y.slice_mut(..));

            y
        }
    }
}

impl<'a, 'b, T>
Sub<Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>> for Scaled<ColVec<T>> where
    T: Gemv + Neg<Output=T>,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Scaled<Product<Transposed<SubMat<T>>, Col<T>>>) -> ColVec<T> {
        unsafe {
            assert_eq_size!(self, rhs);

            let Scaled(beta, mut y) = self;
            let ref beta = beta;
            let Scaled(alpha, Product(Transposed(a), x)) = rhs;
            let ref alpha = alpha.neg();

            let ref trans = Transpose::Yes;

            ops::gemv(trans, alpha, a, beta, x, y.slice_mut(..));

            y
        }
    }
}

impl<'a, 'b, T>
Sub<Scaled<Product<SubMat<'a, T>, Col<'b, T>>>> for Scaled<ColVec<T>> where
    T: Gemv + Neg<Output=T>,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Scaled<Product<SubMat<T>, Col<T>>>) -> ColVec<T> {
        unsafe {
            assert_eq_size!(self, rhs);

            let Scaled(beta, mut y) = self;
            let ref beta = beta;
            let Scaled(alpha, Product(a, x)) = rhs;
            let ref alpha = alpha.neg();

            let ref trans = Transpose::No;

            ops::gemv(trans, alpha, a, beta, x, y.slice_mut(..));

            y
        }
    }
}

impl<'a, 'b, T> Sub<Scaled<ColVec<T>>> for Scaled<Product<Chain<'a, T>, Col<'b, T>>> where
    T: Gemm + Gemv + Neg<Output=T> + One + Zero,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        unsafe {
            use ops::reduce::MatMulCol::*;

            assert_eq_size!(self, rhs);

            let Scaled(beta, lhs) = self;
            let ref beta = beta.neg();
            let Scaled(alpha, mut y) = rhs;
            let ref alpha = alpha;

            let a_mul_b = lhs.reduce();

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

            ops::gemv(transa, alpha, a, beta, x, y.slice_mut(..));

            y
        }
    }
}

impl<'a, 'b, T>
Sub<Scaled<ColVec<T>>> for Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> where
    T: Gemv + Neg<Output=T>,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        unsafe {
            assert_eq_size!(self, rhs);

            let Scaled(alpha, Product(Transposed(a), x)) = self;
            let ref alpha = alpha;
            let Scaled(beta, mut y) = rhs;
            let ref beta = beta.neg();

            let ref trans = Transpose::Yes;

            ops::gemv(trans, alpha, a, beta, x, y.slice_mut(..));

            y
        }
    }
}

impl<'a, 'b, T> Sub<Scaled<ColVec<T>>> for Scaled<Product<SubMat<'a, T>, Col<'b, T>>> where
    T: Gemv + Neg<Output=T>,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        unsafe {
            assert_eq_size!(self, rhs);

            let Scaled(alpha, Product(a, x)) = self;
            let ref alpha = alpha;
            let Scaled(beta, mut y) = rhs;
            let ref beta = beta.neg();

            let ref trans = Transpose::No;

            ops::gemv(trans, alpha, a, beta, x, y.slice_mut(..));

            y
        }
    }
}

// 12 impls
// Secondary implementations
impl<'a, 'b, T> Sub<ColVec<T>> for Product<Chain<'a, T>, Col<'b, T>> where
    T: Gemm + Gemv + Neg<Output=T> + One + Zero,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: ColVec<T>) -> ColVec<T> {
        Scaled(T::one(), self) - Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> Sub<ColVec<T>> for Product<Transposed<SubMat<'a, T>>, Col<'b, T>> where
    T: Gemv + Neg<Output=T> + One,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: ColVec<T>) -> ColVec<T> {
        Scaled(T::one(), self) - Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> Sub<ColVec<T>> for Product<SubMat<'a, T>, Col<'b, T>> where
    T: Gemv + Neg<Output=T> + One,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: ColVec<T>) -> ColVec<T> {
        Scaled(T::one(), self) - Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> Sub<Scaled<ColVec<T>>> for Product<Chain<'a, T>, Col<'b, T>> where
    T: Gemm + Gemv + Neg<Output=T> + One + Zero,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        Scaled(T::one(), self) - rhs
    }
}

impl<'a, 'b, T> Sub<Scaled<ColVec<T>>> for Product<Transposed<SubMat<'a, T>>, Col<'b, T>> where
    T: Gemv + Neg<Output=T> + One,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        Scaled(T::one(), self) - rhs
    }
}

impl<'a, 'b, T> Sub<Scaled<ColVec<T>>> for Product<SubMat<'a, T>, Col<'b, T>> where
    T: Gemv + Neg<Output=T> + One,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        Scaled(T::one(), self) - rhs
    }
}

impl<'a, 'b, T> Sub<Product<Chain<'a, T>, Col<'b, T>>> for Scaled<ColVec<T>> where
    T: Gemm + Gemv + Neg<Output=T> + One + Zero,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Product<Chain<T>, Col<T>>) -> ColVec<T> {
        self - Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> Sub<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> for Scaled<ColVec<T>> where
    T: Gemv + Neg<Output=T> + One,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Product<Transposed<SubMat<T>>, Col<T>>) -> ColVec<T> {
        self - Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> Sub<Product<SubMat<'a, T>, Col<'b, T>>> for Scaled<ColVec<T>> where
    T: Gemv + Neg<Output=T> + One,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: Product<SubMat<T>, Col<T>>) -> ColVec<T> {
        self - Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> Sub<ColVec<T>> for Scaled<Product<Chain<'a, T>, Col<'b, T>>> where
    T: Gemm + Gemv + Neg<Output=T> + One + Zero,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: ColVec<T>) -> ColVec<T> {
        self - Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> Sub<ColVec<T>> for Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> where
    T: Gemv + Neg<Output=T> + One,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: ColVec<T>) -> ColVec<T> {
        self - Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> Sub<ColVec<T>> for Scaled<Product<SubMat<'a, T>, Col<'b, T>>> where
    T: Gemv + Neg<Output=T> + One,
{
    type Output = ColVec<T>;

    fn sub(self, rhs: ColVec<T>) -> ColVec<T> {
        self - Scaled(T::one(), rhs)
    }
}

// 6 impls
assign!(ColVec<T> {
    Product<Chain<'a, T>, Col<'b, T>> { Gemm, Gemv, One, Zero },
    Product<Transposed<SubMat<'a, T>>, Col<'b, T>> { Gemv, One },
    Product<SubMat<'a, T>, Col<'b, T>> { Gemv, One },
    Scaled<Product<Chain<'a, T>, Col<'b, T>>> { Gemm, Gemv, One, Zero },
    Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> { Gemv, One },
    Scaled<Product<SubMat<'a, T>, Col<'b, T>>> { Gemv, One },
});

// AXPY
// Combinations:
//
// LHS: ColVec
// RHS: &T, T, Col, &ColMut, &ColVec, Scaled<Col>
//
// -> 6 implementations

// 6 impls
assign!(ColVec<T> {
    &'a T { Axpy, One },
    T { Axpy, One },
    Col<'a, T> { Axpy, One },
    &'a ColMut<'b, T> { Axpy, One },
    &'a ColVec<T> { Axpy, One },
    Scaled<Col<'a, T>> { Axpy },
});
