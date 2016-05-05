use std::ops::AddAssign;

use blas::{Axpy, Gemm, Transpose};
use onezero::{One, Zero};

use Forward;
use ops::{Reduce, self};
use traits::{
    Matrix, MatrixCols, MatrixColsMut, MatrixRows, MatrixRowsMut, Slice, SliceMut,
};
use traits::Transpose as _0;
use {Chain, ColMut, Col, Mat, RowMut, Row, Scaled, Transposed, SubMat, SubMatMut};

// Combinations:
//
// LHS: Mat, Transposed<Mat>, Transposed<SubMatMut>, SubMatMut
// RHS: &T, T, Chain, &Mat, Scaled<Chain>, Scaled<Transposed<SubMat>>, Scaled<SubMat>,
// &Transposed<Mat>, Transposed<SubMat>, &Transposed<SubMatMut>, SubMat, &SubMatMut
//
// -> 48 implementations

// 4 impls
// Core implementations
impl<'a, 'b, T> AddAssign<&'a T> for SubMatMut<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        let ref alpha = T::one();
        let x = rhs;

        if let Some(y) = self.as_slice_mut() {
            return ops::axpy_slice_scalar(alpha, rhs, y)
        }

        if self.nrows() < self.ncols() {
            for RowMut(Row(ref mut y)) in self.rows_mut() {
                ops::axpy_strided_scalar(alpha, x, y)
            }
        } else {
            for ColMut(Col(ref mut y)) in self.cols_mut() {
                ops::axpy_strided_scalar(alpha, x, y)
            }
        }
    }
}

impl<'a, 'b, T> AddAssign<Scaled<Chain<'a, T>>> for SubMatMut<'b, T> where T: Gemm + One + Zero {
    fn add_assign(&mut self, rhs: Scaled<Chain<T>>) {
        unsafe {
            use ops::reduce::MatMulMat::*;

            assert_eq_size!(self, rhs);

            let Scaled(alpha, chain) = rhs;
            let ref alpha = alpha;
            let ref beta = T::one();
            let c = SubMatMut(self.0);

            let a_mul_b = chain.reduce();

            let (ref transa, ref transb, a, b) = match a_mul_b {
                M_M(ref lhs, ref rhs) => {
                    (Transpose::No, Transpose::No, lhs.slice(..), rhs.slice(..))
                },
                M_SM(ref lhs, (transb, b)) => {
                    (Transpose::No, transb, lhs.slice(..), b)
                },
                SM_M((transa, a), ref rhs) => {
                    (transa, Transpose::No, a, rhs.slice(..))
                },
                SM_SM((transa, a), (transb, b)) => {
                    (transa, transb, a, b)
                },
            };

            ops::gemm(transa, transb, alpha, a, b, beta, c);
        }
    }
}

impl<'a, 'b, T> AddAssign<Scaled<Transposed<SubMat<'a, T>>>> for SubMatMut<'b, T> where T: Axpy {
    fn add_assign(&mut self, rhs: Scaled<Transposed<SubMat<T>>>) {
        unsafe {
            assert_eq_size!(self, rhs);

            let Scaled(ref alpha, rhs) = rhs;

            if self.nrows() < self.ncols() {
                for (RowMut(Row(ref mut y)), Row(ref x)) in self.rows_mut().zip(rhs.rows()) {
                    ops::axpy_strided_strided(alpha, x, y)
                }
            } else {
                for (ColMut(Col(ref mut y)), Col(ref x)) in self.cols_mut().zip(rhs.cols()) {
                    ops::axpy_strided_strided(alpha, x, y)
                }
            }
        }
    }
}

impl<'a, 'b, T> AddAssign<Scaled<SubMat<'a, T>>> for SubMatMut<'b, T> where T: Axpy {
    fn add_assign(&mut self, rhs: Scaled<SubMat<T>>) {
        unsafe {
            assert_eq_size!(self, rhs);

            let Scaled(ref alpha, rhs) = rhs;

            if let (Some(y), Some(x)) = (self.as_slice_mut(), rhs.as_slice()) {
                return ops::axpy_slice_slice(alpha, x, y)
            }

            if self.nrows() < self.ncols() {
                for (RowMut(Row(ref mut y)), Row(ref x)) in self.rows_mut().zip(rhs.rows()) {
                    ops::axpy_strided_strided(alpha, x, y)
                }
            } else {
                for (ColMut(Col(ref mut y)), Col(ref x)) in self.cols_mut().zip(rhs.cols()) {
                    ops::axpy_strided_strided(alpha, x, y)
                }
            }
        }
    }
}

// 10 impls
// Secondary implementations
impl<'a, 'b, T> AddAssign<&'a T> for Transposed<SubMatMut<'b, T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        self.0 += rhs
    }
}

impl<'a, 'b, T> AddAssign<Chain<'a, T>> for Transposed<SubMatMut<'b, T>> where
    T: Gemm + One + Zero,
{
    fn add_assign(&mut self, rhs: Chain<T>) {
        *self += Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> AddAssign<Scaled<Chain<'a, T>>> for Transposed<SubMatMut<'b, T>> where
    T: Gemm + One + Zero,
{
    fn add_assign(&mut self, rhs: Scaled<Chain<T>>) {
        self.0 += rhs.t()
    }
}

impl<'a, 'b, T> AddAssign<Scaled<Transposed<SubMat<'a, T>>>> for Transposed<SubMatMut<'b, T>> where
    T: Axpy,
{
    fn add_assign(&mut self, rhs: Scaled<Transposed<SubMat<'a, T>>>) {
        self.0 += Scaled(rhs.0, (rhs.1).0)
    }
}

impl<'a, 'b, T> AddAssign<Scaled<SubMat<'a, T>>> for Transposed<SubMatMut<'b, T>> where T: Axpy {
    fn add_assign(&mut self, rhs: Scaled<SubMat<'a, T>>) {
        self.0 += rhs.t()
    }
}

impl<'a, 'b, T> AddAssign<Transposed<SubMat<'a, T>>> for Transposed<SubMatMut<'b, T>> where
    T: Axpy + One,
{
    fn add_assign(&mut self, rhs: Transposed<SubMat<'a, T>>) {
        *self += Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> AddAssign<SubMat<'a, T>> for Transposed<SubMatMut<'b, T>> where
    T: Axpy + One,
{
    fn add_assign(&mut self, rhs: SubMat<'a, T>) {
        *self += Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> AddAssign<Chain<'a, T>> for SubMatMut<'b, T> where T: Gemm + One + Zero {
    fn add_assign(&mut self, rhs: Chain<T>) {
        *self += Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> AddAssign<Transposed<SubMat<'a, T>>> for SubMatMut<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Transposed<SubMat<T>>) {
        *self += Scaled(T::one(), rhs)
    }
}

impl<'a, 'b, T> AddAssign<SubMat<'a, T>> for SubMatMut<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: SubMat<T>) {
        *self += Scaled(T::one(), rhs)
    }
}

macro_rules! forward {
    ($lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, 'c, T> AddAssign<$rhs> for $lhs where $(T: $bound),+ {
                fn add_assign(&mut self, rhs: $rhs) {
                    self.slice_mut(..) += rhs.slice(..)
                }
            }
         )+
    };
}

// 12 impls
forward!(Mat<T> {
    Chain<'a, T> { Gemm, One, Zero },
    &'a Mat<T> { Axpy, One },
    Scaled<Chain<'a, T>> { Gemm, One, Zero },
    Scaled<Transposed<SubMat<'a, T>>> { Axpy },
    Scaled<SubMat<'a, T>> { Axpy },
    &'a Transposed<Mat<T>> { Axpy, One },
    Transposed<SubMat<'a, T>> { Axpy, One },
    &'a Transposed<SubMatMut<'b, T>> { Axpy, One },
    SubMat<'a, T> { Axpy, One },
    &'a SubMatMut<'b, T> { Axpy, One },
});

impl<'a, T> AddAssign<&'a T> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        self.slice_mut(..) += rhs
    }
}

impl<T> AddAssign<T> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        self.slice_mut(..) += &rhs
    }
}

// 12 impls
forward!(Transposed<Mat<T>> {
    Chain<'a, T> { Gemm, One, Zero },
    &'a Mat<T> { Axpy, One },
    Scaled<Chain<'a, T>> { Gemm, One, Zero },
    Scaled<Transposed<SubMat<'a, T>>> { Axpy },
    Scaled<SubMat<'a, T>> { Axpy },
    &'a Transposed<Mat<T>> { Axpy, One },
    Transposed<SubMat<'a, T>> { Axpy, One },
    &'a Transposed<SubMatMut<'b, T>> { Axpy, One },
    SubMat<'a, T> { Axpy, One },
    &'a SubMatMut<'b, T> { Axpy, One },
});

impl<'a, T> AddAssign<&'a T> for Transposed<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        self.slice_mut(..) += rhs
    }
}

impl<T> AddAssign<T> for Transposed<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        self.slice_mut(..) += &rhs
    }
}

// 5 impls
forward!(Transposed<SubMatMut<'a, T>> {
    &'b Mat<T> { Axpy, One },
    &'b Transposed<Mat<T>> { Axpy, One },
    &'b Transposed<SubMatMut<'c, T>> { Axpy, One },
    &'b SubMatMut<'c, T> { Axpy, One },
});

impl<'a, T> AddAssign<T> for Transposed<SubMatMut<'a, T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        self.0 += &rhs
    }
}

// 5 impls
forward!(SubMatMut<'a, T> {
    &'b Mat<T> { Axpy, One },
    &'b Transposed<Mat<T>> { Axpy, One },
    &'b Transposed<SubMatMut<'c, T>> { Axpy, One },
    &'b SubMatMut<'c, T> { Axpy, One },
});

impl<'a, T> AddAssign<T> for SubMatMut<'a, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        *self += &rhs
    }
}
