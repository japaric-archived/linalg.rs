use std::ops::Add;

use assign::AddAssign;
use blas::{Axpy, Gemm, Gemv, Transpose};
use complex::Complex;
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

// 3 impls
// Core implementations
impl<'a, 'b, T> Add<Scaled<Product<Chain<'a, T>, Col<'b, T>>>> for Scaled<ColVec<T>> where
    T: Gemm + Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Scaled<Product<Chain<T>, Col<T>>>) -> ColVec<T> {
        unsafe {
            use ops::reduce::MatMulCol::*;

            assert_eq_size!(self, rhs);

            let Scaled(alpha, mut y) = self;
            let ref alpha = alpha;
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
Add<Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>> for Scaled<ColVec<T>> where
    T: Gemv,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Scaled<Product<Transposed<SubMat<T>>, Col<T>>>) -> ColVec<T> {
        unsafe {
            assert_eq_size!(self, rhs);

            let ref transa = Transpose::Yes;
            let Scaled(ref alpha, Product(Transposed(a), x)) = rhs;
            let Scaled(beta, mut y) = self;
            let ref beta = beta;

            ops::gemv(transa, alpha, a, beta, x, y.slice_mut(..));

            y
        }
    }
}

impl<'a, 'b, T> Add<Scaled<Product<SubMat<'a, T>, Col<'b, T>>>> for Scaled<ColVec<T>> where
    T: Gemv,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Scaled<Product<SubMat<T>, Col<T>>>) -> ColVec<T> {
        unsafe {
            assert_eq_size!(self, rhs);

            let ref transa = Transpose::No;
            let Scaled(ref alpha, Product(a, x)) = rhs;
            let Scaled(beta, mut y) = self;
            let ref beta = beta;

            ops::gemv(transa, alpha, a, beta, x, y.slice_mut(..));

            y
        }
    }
}

// 3 impls
// Secondary implementations
impl<'a, 'b, T> Add<Product<Chain<'a, T>, Col<'b, T>>> for Scaled<ColVec<T>> where
    T: Gemm + Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Product<Chain<T>, Col<T>>) -> ColVec<T> {
        self + Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> Add<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> for Scaled<ColVec<T>> where
    T: Gemv + One,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Product<Transposed<SubMat<T>>, Col<T>>) -> ColVec<T> {
        self + Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> Add<Product<SubMat<'a, T>, Col<'b, T>>> for Scaled<ColVec<T>> where
    T: Gemv + One,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Product<SubMat<T>, Col<T>>) -> ColVec<T> {
        self + Scaled(T::one(), rhs)
    }
}

// 6 impls
// Reverse impls
impl<'a, 'b, T> Add<Scaled<ColVec<T>>> for Scaled<Product<Chain<'a, T>, Col<'b, T>>> where
    T: Gemm + Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        rhs + self
    }
}

impl<'a, 'b, T>
Add<Scaled<ColVec<T>>> for Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> where
    T: Gemv,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        rhs + self
    }
}

impl<'a, 'b, T> Add<Scaled<ColVec<T>>> for Scaled<Product<SubMat<'a, T>, Col<'b, T>>> where
    T: Gemv,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        rhs + self
    }
}

impl<'a, 'b, T> Add<Scaled<ColVec<T>>> for Product<Chain<'a, T>, Col<'b, T>> where
    T: Gemm + Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        rhs + self
    }
}

impl<'a, 'b, T> Add<Scaled<ColVec<T>>> for Product<Transposed<SubMat<'a, T>>, Col<'b, T>> where
    T: Gemv + One,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        rhs + self
    }
}

impl<'a, 'b, T> Add<Scaled<ColVec<T>>> for Product<SubMat<'a, T>, Col<'b, T>> where
    T: Gemv + One,
{
    type Output = ColVec<T>;

    fn add(self, rhs: Scaled<ColVec<T>>) -> ColVec<T> {
        rhs + self
    }
}

// 12 impls
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
// and the reverse operation
//
// -> 12 implementations

// 10 impls
assign!(half ColVec<T> {
    &'a T { Axpy, One },
    T { Axpy, One },
});

assign!(ColVec<T> {
    Col<'a, T> { Axpy, One },
    &'a ColMut<'b, T> { Axpy, One },
    &'a ColVec<T> { Axpy, One },
    Scaled<Col<'a, T>> { Axpy },
});

macro_rules! scalar {
    ($($t:ty),+) => {
        $(
            impl<'a> Add<ColVec<$t>> for &'a $t {
                type Output = ColVec<$t>;

                fn add(self, mut rhs: ColVec<$t>) -> ColVec<$t> {
                    rhs.add_assign(self);
                    rhs
                }
            }

            impl Add<ColVec<$t>> for $t {
                type Output = ColVec<$t>;

                fn add(self, mut rhs: ColVec<$t>) -> ColVec<$t> {
                    rhs.add_assign(self);
                    rhs
                }
            }
         )+
    };
}

// 2 impls
scalar!(f32, f64, Complex<f32>, Complex<f64>);
