use blas::{Gemm, Gemv, Transpose};
use onezero::{One, Zero};

use ops::{Reduce, self};
use traits::Transpose as _0;
use traits::{Eval, Matrix, Slice, SliceMut};
use {Chain, Col, ColVec, Mat, Product, Row, RowVec, Scaled, Tor, Transposed, SubMat};

/// alpha * op(A) * x
unsafe fn gemv<T>(
    trans: &Transpose,
    alpha: &T,
    a: SubMat<T>,
    x: Col<T>,
) -> ColVec<T> where
    T: Gemv + Zero,
{
    let mut y = ColVec(Tor::uninitialized(match *trans {
        Transpose::No => a.nrows,
        Transpose::Yes => a.ncols,
    }));

    ops::gemv(trans, alpha, a, &T::zero(), x, y.slice_mut(..));

    y
}

// Core implementations
impl<'a, T> Eval for Scaled<Chain<'a, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn eval(self) -> Mat<T> {
        unsafe {
            use ops::reduce::MatMulMat::*;

            let Scaled(alpha, chain) = self;
            let ref alpha = alpha;

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

            ops::submat_mul_submat(transa, transb, alpha, a, b)
        }
    }
}

impl<'a, 'b, T> Eval for Scaled<Product<Chain<'a, T>, Col<'b, T>>> where
    T: Gemm + Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn eval(self) -> ColVec<T> {
        unsafe {
            use ops::reduce::MatMulCol::*;

            let Scaled(alpha, product) = self;
            let ref alpha = alpha;

            let a_mul_b = product.reduce();

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

            ops::submat_mul_col(transa, alpha, a, x)
        }
    }
}

impl<'a, 'b, T> Eval for Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> where
    T: Gemv + Zero,
{
    type Output = ColVec<T>;

    fn eval(self) -> ColVec<T> {
        unsafe {
            let Scaled(ref alpha, Product(Transposed(a), x)) = self;
            let ref trans = Transpose::Yes;

            gemv(trans, alpha, a, x)
        }
    }
}

impl<'a, 'b, T> Eval for Scaled<Product<SubMat<'a, T>, Col<'b, T>>> where T: Gemv + Zero {
    type Output = ColVec<T>;

    fn eval(self) -> ColVec<T> {
        unsafe {
            let Scaled(ref alpha, Product(a, x)) = self;
            let ref trans = Transpose::No;

            gemv(trans, alpha, a, x)
        }
    }
}

// Secondary implementations
impl<'a, T> Eval for Chain<'a, T> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn eval(self) -> Mat<T> {
        Scaled(T::one(), self).eval()
    }
}

impl<'a, 'b, T> Eval for Product<Chain<'a, T>, Col<'b, T>> where T: Gemm + Gemv + One + Zero {
    type Output = ColVec<T>;

    fn eval(self) -> ColVec<T> {
        Scaled(T::one(), self).eval()
    }
}

impl<'a, 'b, T> Eval for Product<Transposed<SubMat<'a, T>>, Col<'b, T>> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn eval(self) -> ColVec<T> {
        Scaled(T::one(), self).eval()
    }
}

impl<'a, 'b, T> Eval for Product<SubMat<'a, T>, Col<'b, T>> where T: Gemv + One + Zero {
    type Output = ColVec<T>;

    fn eval(self) -> ColVec<T> {
        Scaled(T::one(), self).eval()
    }
}

impl<'a, 'b, T> Eval for Product<Row<'a, T>, Chain<'b, T>> where T: Gemm + Gemv + One + Zero {
    type Output = RowVec<T>;

    fn eval(self) -> RowVec<T> {
        Scaled(T::one(), self).eval()
    }
}

impl<'a, 'b, T> Eval for Product<Row<'a, T>, Transposed<SubMat<'b, T>>> where
    T: Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn eval(self) -> RowVec<T> {
        Scaled(T::one(), self).eval()
    }
}

impl<'a, 'b, T> Eval for Product<Row<'a, T>, SubMat<'b, T>> where T: Gemv + One + Zero {
    type Output = RowVec<T>;

    fn eval(self) -> RowVec<T> {
        Scaled(T::one(), self).eval()
    }
}

impl<'a, 'b, T> Eval for Scaled<Product<Row<'a, T>, Chain<'b, T>>> where
    T: Gemm + Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn eval(self) -> RowVec<T> {
        self.t().eval().t()
    }
}

impl<'a, 'b, T> Eval for Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>> where
    T: Gemv + Zero,
{
    type Output = RowVec<T>;

    fn eval(self) -> RowVec<T> {
        self.t().eval().t()
    }
}

impl<'a, 'b, T> Eval for Scaled<Product<Row<'a, T>, SubMat<'b, T>>> where
    T: Gemv + Zero,
{
    type Output = RowVec<T>;

    fn eval(self) -> RowVec<T> {
        self.t().eval().t()
    }
}
