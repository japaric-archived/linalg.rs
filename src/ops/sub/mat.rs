use std::ops::{Neg, Sub};

use blas::{Axpy, Gemm, Transpose};
use onezero::{One, Zero};

use ops::{Reduce, self};
use traits::Transpose as _0;
use traits::{Matrix, SliceMut, Slice};
use {Chain, Mat, Scaled, Transposed, SubMat, SubMatMut};

// GEMM
// Combinations:
//
// LHS: Mat, Scaled<Mat>, Scaled<Transposed<Mat>>, Transposed<Mat>
// RHS: Chain, Scaled<Chain>
//
// and the reverse operation
//
// -> 16 implementations

// 2 impls
// Core implementations
impl<'a, T> Sub<Scaled<Mat<T>>> for Scaled<Chain<'a, T>> where
    T: Gemm + Neg<Output=T> + One + Zero,
{
    type Output = Mat<T>;

    fn sub(self, rhs: Scaled<Mat<T>>) -> Mat<T> {
        unsafe {
            use ops::reduce::MatMulMat::*;

            assert_eq_size!(self, rhs);

            let Scaled(alpha, chain) = self;
            let ref alpha = alpha;
            let Scaled(beta, mut c) = rhs;
            let ref beta = beta.neg();

            let a_mul_b = chain.reduce();

            let (ref transa, ref transb, a, b) = match a_mul_b {
                M_M(ref lhs, ref rhs) => {
                    (Transpose::No, Transpose::No, lhs.slice(..), rhs.slice(..))
                },
                M_SM(ref lhs, (transb, b)) => {
                    (Transpose::No, transb, lhs.slice(..), b)
                }
                SM_M((transa, a), ref rhs) => {
                    (transa, Transpose::No, a, rhs.slice(..))
                }
                SM_SM((transa, a), (transb, b)) => {
                    (transa, transb, a, b)
                }
            };

            ops::gemm(transa, transb, alpha, a, b, beta, c.slice_mut(..));

            c
        }
    }
}

impl<'a, T> Sub<Scaled<Chain<'a, T>>> for Scaled<Mat<T>> where
    T: Gemm + Neg<Output=T> + One + Zero,
{
    type Output = Mat<T>;

    fn sub(self, rhs: Scaled<Chain<T>>) -> Mat<T> {
        unsafe {
            use ops::reduce::MatMulMat::*;

            assert_eq_size!(self, rhs);

            let Scaled(beta, mut c) = self;
            let ref beta = beta;
            let Scaled(alpha, chain) = rhs;
            let ref alpha = alpha.neg();

            let a_mul_b = chain.reduce();

            let (ref transa, ref transb, a, b) = match a_mul_b {
                M_M(ref lhs, ref rhs) => {
                    (Transpose::No, Transpose::No, lhs.slice(..), rhs.slice(..))
                },
                M_SM(ref lhs, (transb, b)) => {
                    (Transpose::No, transb, lhs.slice(..), b)
                }
                SM_M((transa, a), ref rhs) => {
                    (transa, Transpose::No, a, rhs.slice(..))
                }
                SM_SM((transa, a), (transb, b)) => {
                    (transa, transb, a, b)
                }
            };

            ops::gemm(transa, transb, alpha, a, b, beta, c.slice_mut(..));

            c
        }
    }
}

// 10 impls
// Secondary implementations
impl<'a, T> Sub<Mat<T>> for Chain<'a, T> where T: Gemm + Neg<Output=T> + One + Zero {
    type Output = Mat<T>;

    fn sub(self, rhs: Mat<T>) -> Mat<T> {
        Scaled(T::one(), self) - Scaled(T::one(), rhs)
    }
}

impl<'a, T> Sub<Transposed<Mat<T>>> for Chain<'a, T> where T: Gemm + Neg<Output=T> + One + Zero {
    type Output = Transposed<Mat<T>>;

    fn sub(self, rhs: Transposed<Mat<T>>) -> Transposed<Mat<T>> {
        Scaled(T::one(), self) - Scaled(T::one(), rhs)
    }
}

impl<'a, T> Sub<Scaled<Mat<T>>> for Chain<'a, T> where T: Gemm + Neg<Output=T> + One + Zero {
    type Output = Mat<T>;

    fn sub(self, rhs: Scaled<Mat<T>>) -> Mat<T> {
        Scaled(T::one(), self) - rhs
    }
}

impl<'a, T> Sub<Scaled<Transposed<Mat<T>>>> for Chain<'a, T> where
    T: Gemm + Neg<Output=T> + One + Zero,
{
    type Output = Transposed<Mat<T>>;

    fn sub(self, rhs: Scaled<Transposed<Mat<T>>>) -> Transposed<Mat<T>> {
        Scaled(T::one(), self) - rhs
    }
}

impl<'a, T> Sub<Mat<T>> for Scaled<Chain<'a, T>> where T: Gemm + Neg<Output=T> + One + Zero {
    type Output = Mat<T>;

    fn sub(self, rhs: Mat<T>) -> Mat<T> {
        self - Scaled(T::one(), rhs)
    }
}

impl<'a, T> Sub<Scaled<Transposed<Mat<T>>>> for Scaled<Chain<'a, T>>
    where T: Gemm + Neg<Output=T> + One + Zero,
{
    type Output = Transposed<Mat<T>>;

    fn sub(self, rhs: Scaled<Transposed<Mat<T>>>) -> Transposed<Mat<T>> {
        (self.t() - rhs.t()).t()
    }
}

impl<'a, T> Sub<Transposed<Mat<T>>> for Scaled<Chain<'a, T>>
    where T: Gemm + Neg<Output=T> + One + Zero,
{
    type Output = Transposed<Mat<T>>;

    fn sub(self, rhs: Transposed<Mat<T>>) -> Transposed<Mat<T>> {
        self - Scaled(T::one(), rhs)
    }
}

impl<'a, T> Sub<Chain<'a, T>> for Scaled<Mat<T>> where T: Gemm + Neg<Output=T> + One + Zero {
    type Output = Mat<T>;

    fn sub(self, rhs: Chain<T>) -> Mat<T> {
        self - Scaled(T::one(), rhs)
    }
}

impl<'a, T> Sub<Chain<'a, T>> for Scaled<Transposed<Mat<T>>> where
    T: Gemm + Neg<Output=T> + One + Zero,
{
    type Output = Transposed<Mat<T>>;

    fn sub(self, rhs: Chain<T>) -> Transposed<Mat<T>> {
        self - Scaled(T::one(), rhs)
    }
}

impl<'a, T> Sub<Scaled<Chain<'a, T>>> for Scaled<Transposed<Mat<T>>> where
    T: Gemm + Neg<Output=T> + One + Zero,
{
    type Output = Transposed<Mat<T>>;

    fn sub(self, rhs: Scaled<Chain<T>>) -> Transposed<Mat<T>> {
        (self.t() - rhs.t()).t()
    }
}

// 2 impls
assign!(Mat<T> {
    Chain<'a, T> { Gemm, One, Zero },
    Scaled<Chain<'a, T>> { Gemm, One, Zero },
});

// 2 impls
assign!(Transposed<Mat<T>> {
    Chain<'a, T> { Gemm, One, Zero },
    Scaled<Chain<'a, T>> { Gemm, One, Zero },
});

// AXPY
// Combinations:
//
// LHS: Mat, Transposed<Mat>
// RHS: &T, T, &Mat, Scaled<Transposed<SubMat>>, Scaled<SubMat>, &Transposed<Mat>,
// Transposed<SubMat>,
//      &Transposed<SubMatMut>, SubMat, &SubMatMut
//
// -> 20 implementations

// 10 impls
assign!(Mat<T> {
    &'a T { Axpy, One },
    T { Axpy, One },
    &'a Mat<T> { Axpy, One },
    Scaled<Transposed<SubMat<'a, T>>> { Axpy },
    Scaled<SubMat<'a, T>> { Axpy },
    &'a Transposed<Mat<T>> { Axpy, One },
    Transposed<SubMat<'a, T>> { Axpy, One },
    &'a Transposed<SubMatMut<'b, T>> { Axpy, One },
    SubMat<'a, T> { Axpy, One },
    &'a SubMatMut<'b, T> { Axpy, One },
});

// 10 impls
assign!(Transposed<Mat<T>> {
    &'a T { Axpy, One },
    T { Axpy, One },
    &'a Mat<T> { Axpy, One },
    Scaled<Transposed<SubMat<'a, T>>> { Axpy },
    Scaled<SubMat<'a, T>> { Axpy },
    &'a Transposed<Mat<T>> { Axpy, One },
    Transposed<SubMat<'a, T>> { Axpy, One },
    &'a Transposed<SubMatMut<'b, T>> { Axpy, One },
    SubMat<'a, T> { Axpy, One },
    &'a SubMatMut<'b, T> { Axpy, One },
});
