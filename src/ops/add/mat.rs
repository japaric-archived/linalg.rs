use std::ops::Add;

use blas::{Axpy, Gemm, Transpose};
use complex::Complex;
use onezero::{One, Zero};

use ops::{Reduce, self};
use traits::Transpose as _0;
use traits::{Matrix, Slice, SliceMut};
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

// 1 impls
// Core implementations
impl<'a, T> Add<Scaled<Chain<'a, T>>> for Scaled<Mat<T>> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn add(self, rhs: Scaled<Chain<T>>) -> Mat<T> {
        unsafe {
            use ops::reduce::MatMulMat::*;

            assert_eq_size!(self, rhs);

            let Scaled(beta, mut c) = self;
            let ref beta = beta;
            let Scaled(alpha, chain) = rhs;
            let ref alpha = alpha;

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

// 3 impls
// Secondary implementations
impl<'a, T> Add<Chain<'a, T>> for Scaled<Mat<T>> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn add(self, rhs: Chain<T>) -> Mat<T> {
        self + Scaled(T::one(), rhs)
    }
}

impl<'a, T> Add<Scaled<Chain<'a, T>>> for Scaled<Transposed<Mat<T>>> where T: Gemm + One + Zero {
    type Output = Transposed<Mat<T>>;

    fn add(self, rhs: Scaled<Chain<T>>) -> Transposed<Mat<T>> {
        (self.t() + rhs.t()).t()
    }
}

impl<'a, T> Add<Chain<'a, T>> for Scaled<Transposed<Mat<T>>> where T: Gemm + One + Zero {
    type Output = Transposed<Mat<T>>;

    fn add(self, rhs: Chain<T>) -> Transposed<Mat<T>> {
        self + Scaled(T::one(), rhs)
    }
}

// 4 impls
// Reverse implementations
impl<'a, T> Add<Scaled<Mat<T>>> for Scaled<Chain<'a, T>> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn add(self, rhs: Scaled<Mat<T>>) -> Mat<T> {
        rhs + self
    }
}

impl<'a, T> Add<Scaled<Mat<T>>> for Chain<'a, T> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn add(self, rhs: Scaled<Mat<T>>) -> Mat<T> {
        rhs + self
    }
}

impl<'a, T> Add<Scaled<Transposed<Mat<T>>>> for Scaled<Chain<'a, T>> where T: Gemm + One + Zero {
    type Output = Transposed<Mat<T>>;

    fn add(self, rhs: Scaled<Transposed<Mat<T>>>) -> Transposed<Mat<T>> {
        rhs + self
    }
}

impl<'a, T> Add<Scaled<Transposed<Mat<T>>>> for Chain<'a, T> where T: Gemm + One + Zero {
    type Output = Transposed<Mat<T>>;

    fn add(self, rhs: Scaled<Transposed<Mat<T>>>) -> Transposed<Mat<T>> {
        rhs + self
    }
}

// 4 impls
assign!(Mat<T> {
    Chain<'a, T> { Gemm, One, Zero },
    Scaled<Chain<'a, T>> { Gemm, One, Zero },
});

// 4 impls
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
// and the reverse operation
//
// -> 40 implementations

// 18 impls
assign!(half Mat<T> {
    &'a T { Axpy, One },
    T { Axpy, One },
});

assign!(Mat<T> {
    &'a Mat<T> { Axpy, One },
    Scaled<Transposed<SubMat<'a, T>>> { Axpy },
    Scaled<SubMat<'a, T>> { Axpy },
    &'a Transposed<Mat<T>> { Axpy, One },
    &'a Transposed<SubMatMut<'b, T>> { Axpy, One },
    Transposed<SubMat<'a, T>> { Axpy, One },
    SubMat<'a, T> { Axpy, One },
    &'a SubMatMut<'b, T> { Axpy, One },
});

// 18 impls
assign!(half Transposed<Mat<T>> {
    &'a T { Axpy, One },
    T { Axpy, One },
});

assign!(Transposed<Mat<T>> {
    &'a Mat<T> { Axpy, One },
    Scaled<Transposed<SubMat<'a, T>>> { Axpy },
    Scaled<SubMat<'a, T>> { Axpy },
    &'a Transposed<Mat<T>> { Axpy, One },
    &'a Transposed<SubMatMut<'b, T>> { Axpy, One },
    Transposed<SubMat<'a, T>> { Axpy, One },
    SubMat<'a, T> { Axpy, One },
    &'a SubMatMut<'b, T> { Axpy, One },
});

macro_rules! scalar {
    ($($t:ty),+) => {
        $(
            impl<'a> Add<Mat<$t>> for &'a $t {
                type Output = Mat<$t>;

                fn add(self, mut rhs: Mat<$t>) -> Mat<$t> {
                    rhs += self;
                    rhs
                }
            }

            impl Add<Mat<$t>> for $t {
                type Output = Mat<$t>;

                fn add(self, mut rhs: Mat<$t>) -> Mat<$t> {
                    rhs += self;
                    rhs
                }
            }

            impl<'a> Add<Transposed<Mat<$t>>> for &'a $t {
                type Output = Transposed<Mat<$t>>;

                fn add(self, mut rhs: Transposed<Mat<$t>>) -> Transposed<Mat<$t>> {
                    rhs += self;
                    rhs
                }
            }

            impl Add<Transposed<Mat<$t>>> for $t {
                type Output = Transposed<Mat<$t>>;

                fn add(self, mut rhs: Transposed<Mat<$t>>) -> Transposed<Mat<$t>> {
                    rhs += self;
                    rhs
                }
            }
         )+
    };
}

// 4 impls
scalar!(f32, f64, Complex<f32>, Complex<f64>);
