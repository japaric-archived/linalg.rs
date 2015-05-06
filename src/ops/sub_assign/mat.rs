use std::ops::Neg;

use assign::SubAssign;
use blas::{Axpy, Gemm, Transpose};
use onezero::{One, Zero};

use Forward;
use ops::{Reduce, self};
use traits::{
    Matrix, MatrixCols, MatrixColsMut, MatrixRows, MatrixRowsMut, Slice, SliceMut,
};
use traits::Transpose as _0;
use {Chain, Col, ColMut, Mat, Scaled, Row, RowMut, Transposed, SubMat, SubMatMut};

// Combinations:
//
// LHS: Mat, Transposed<Mat>, Transposed<SubMatMut>, SubMatMut
// RHS: &T, T, Chain, &Mat, Scaled<Chain>, Scaled<Transposed<SubMat>>, Scaled<SubMat>,
// &Transposed<Mat>, Transposed<SubMat>, &Transposed<SubMatMut>, SubMat, &SubMatMut
//
// -> 48 implementations

// 4 impls
// Core implementations
impl<'a, 'b, T> SubAssign<&'a T> for SubMatMut<'b, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        let ref alpha = T::one().neg();
        let x = rhs;

        if let Some(y) = self.as_slice_mut() {
            return ops::axpy_slice_scalar(alpha, x, y)
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

impl<'a, 'b, T> SubAssign<Scaled<Chain<'a, T>>> for SubMatMut<'b, T> where
    T: Gemm + Neg<Output=T> + One + Zero,
{
    fn sub_assign(&mut self, rhs: Scaled<Chain<T>>) {
        unsafe {
            use ops::reduce::MatMulMat::*;

            assert_eq_size!(self, rhs);

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

            let c = SubMatMut(self.0);
            let ref beta = T::one();

            ops::gemm(transa, transb, alpha, a, b, beta, c);
        }
    }
}

impl<'a, 'b, T> SubAssign<Scaled<Transposed<SubMat<'a, T>>>> for SubMatMut<'b, T> where
    T: Axpy + Neg<Output=T>,
{
    fn sub_assign(&mut self, rhs: Scaled<Transposed<SubMat<T>>>) {
        unsafe {
            assert_eq_size!(self, rhs);

            let Scaled(alpha, rhs) = rhs;
            let ref alpha = alpha.neg();

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

impl<'a, 'b, T>
SubAssign<Scaled<SubMat<'a, T>>> for SubMatMut<'b, T> where T: Axpy + Neg<Output=T> {
    fn sub_assign(&mut self, rhs: Scaled<SubMat<T>>) {
        unsafe {
            assert_eq_size!(self, rhs);

            let Scaled(alpha, rhs) = rhs;
            let ref alpha = alpha.neg();

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
impl<'a, 'b, T> SubAssign<&'a T> for Transposed<SubMatMut<'b, T>> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &T) {
        self.0.sub_assign(rhs)
    }
}

impl<'a, 'b, T> SubAssign<Chain<'a, T>> for Transposed<SubMatMut<'b, T>> where
    T: Gemm + Neg<Output=T> + One + Zero,
{
    fn sub_assign(&mut self, rhs: Chain<T>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, T> SubAssign<Scaled<Chain<'a, T>>> for Transposed<SubMatMut<'b, T>> where
    T: Gemm + Neg<Output=T> + One + Zero,
{
    fn sub_assign(&mut self, rhs: Scaled<Chain<T>>) {
        self.0.sub_assign(rhs.t())
    }
}

impl<'a, 'b, T> SubAssign<Scaled<Transposed<SubMat<'a, T>>>> for Transposed<SubMatMut<'b, T>> where
    T: Axpy + Neg<Output=T>,
{
    fn sub_assign(&mut self, rhs: Scaled<Transposed<SubMat<'a, T>>>) {
        self.0.sub_assign(Scaled(rhs.0, (rhs.1).0))
    }
}

impl<'a, 'b, T> SubAssign<Scaled<SubMat<'a, T>>> for Transposed<SubMatMut<'b, T>> where
    T: Axpy + Neg<Output=T>,
{
    fn sub_assign(&mut self, rhs: Scaled<SubMat<'a, T>>) {
        self.0.sub_assign(rhs.t())
    }
}

impl<'a, 'b, T> SubAssign<Transposed<SubMat<'a, T>>> for Transposed<SubMatMut<'b, T>> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Transposed<SubMat<'a, T>>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, T> SubAssign<SubMat<'a, T>> for Transposed<SubMatMut<'b, T>> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: SubMat<'a, T>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, T> SubAssign<Chain<'a, T>> for SubMatMut<'b, T> where T:
    Gemm + Neg<Output=T> + One + Zero,
{
    fn sub_assign(&mut self, rhs: Chain<T>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, T> SubAssign<Transposed<SubMat<'a, T>>> for SubMatMut<'b, T> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: Transposed<SubMat<T>>) {
        self.sub_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, T> SubAssign<SubMat<'a, T>> for SubMatMut<'b, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: SubMat<T>) {
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
    };
}

// 12 impls
forward!(Mat<T> {
    Chain<'a, T> { Gemm, One, Zero },
    &'a Mat<T> { Axpy, One },
    Scaled<Chain<'a, T>> { Gemm, One, Zero  },
    Scaled<Transposed<SubMat<'a, T>>> { Axpy  },
    Scaled<SubMat<'a, T>> { Axpy  },
    &'a Transposed<Mat<T>> { Axpy, One },
    Transposed<SubMat<'a, T>> { Axpy, One },
    &'a Transposed<SubMatMut<'b, T>> { Axpy, One },
    SubMat<'a, T> { Axpy, One },
    &'a SubMatMut<'b, T> { Axpy, One },
});

impl<'a, T> SubAssign<&'a T> for Mat<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        self.slice_mut(..).sub_assign(rhs)
    }
}

impl<T> SubAssign<T> for Mat<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.slice_mut(..).sub_assign(&rhs)
    }
}

// 12 impls
forward!(Transposed<Mat<T>> {
    Chain<'a, T> { Gemm, One, Zero },
    &'a Mat<T> { Axpy, One },
    Scaled<Chain<'a, T>> { Gemm, One, Zero },
    Scaled<Transposed<SubMat<'a, T>>> { Axpy  },
    Scaled<SubMat<'a, T>> { Axpy  },
    &'a Transposed<Mat<T>> { Axpy, One },
    Transposed<SubMat<'a, T>> { Axpy, One },
    &'a Transposed<SubMatMut<'b, T>> { Axpy, One },
    SubMat<'a, T> { Axpy, One },
    &'a SubMatMut<'b, T> { Axpy, One },
});

impl<'a, T> SubAssign<&'a T> for Transposed<Mat<T>> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        self.slice_mut(..).sub_assign(rhs)
    }
}

impl<T> SubAssign<T> for Transposed<Mat<T>> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.slice_mut(..).sub_assign(&rhs)
    }
}

// 5 impls
forward!(Transposed<SubMatMut<'a, T>> {
    &'b Mat<T> { Axpy, One },
    &'b Transposed<Mat<T>> { Axpy, One },
    &'b Transposed<SubMatMut<'c, T>> { Axpy, One },
    &'b SubMatMut<'c, T> { Axpy, One },
});

impl<'a, T> SubAssign<T> for Transposed<SubMatMut<'a, T>> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.0.sub_assign(&rhs)
    }
}

// 5 impls
forward!(SubMatMut<'a, T> {
    &'b Mat<T> { Axpy, One },
    &'b Transposed<Mat<T>> { Axpy, One },
    &'b Transposed<SubMatMut<'c, T>> { Axpy, One },
    &'b SubMatMut<'c, T> { Axpy, One },
});

impl<'a, T> SubAssign<T> for SubMatMut<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.sub_assign(&rhs)
    }
}
