use onezero::{One, Zero};
use std::num::Int;

use {Col, Mat, MutView, Row, Trans, View};
use blas::{Dot, Gemm, Gemv, ToBlasInt, Transpose, Vector, mod};
use traits::{Collection, Matrix};

impl<T, L, R> Mul<Col<R>, T> for Row<L> where T: Dot + Zero, L: Vector<T>, R: Vector<T> {
    fn mul(&self, rhs: &Col<R>) -> T {
        assert_eq!(self.len(), rhs.len());

        let n = Collection::len(&self.0);

        if n == 0 { return Zero::zero() }

        let dot = Dot::dot(None::<T>);
        let n = n.to_blasint();
        let x = self.0.as_ptr();
        let incx = self.0.stride();
        let y = rhs.0.as_ptr();
        let incy = rhs.0.stride();

        unsafe {
            dot(&n, x, &incx, y, &incy)
        }
    }
}

impl<T, V> Mul<Col<V>, Col<Box<[T]>>> for Mat<T> where T: Gemv + One + Zero, V: Vector<T> {
    fn mul(&self, rhs: &Col<V>) -> Col<Box<[T]>> {
        mc(self, rhs)
    }
}

impl<T, V> Mul<Col<V>, Col<Box<[T]>>> for Trans<Mat<T>> where T: Gemv + One + Zero, V: Vector<T> {
    fn mul(&self, rhs: &Col<V>) -> Col<Box<[T]>> {
        mc(self, rhs)
    }
}

impl<T, M> Mul<M, Mat<T>> for Mat<T> where T: Gemm + One + Zero, M: blas::Matrix<T> {
    fn mul(&self, rhs: &M) -> Mat<T> {
        mm(self, rhs)
    }
}

impl<T, M> Mul<M, Mat<T>> for Trans<Mat<T>> where T: Gemm + One + Zero, M: blas::Matrix<T> {
    fn mul(&self, rhs: &M) -> Mat<T> {
        mm(self, rhs)
    }
}

impl<T, V, M> Mul<M, Row<Box<[T]>>> for Row<V> where
    T: Gemv + One + Zero,
    M: blas::Matrix<T>,
    V: Vector<T>,
{
    fn mul(&self, rhs: &M) -> Row<Box<[T]>> {
        rm(self, rhs)
    }
}

macro_rules! view {
    ($($ty:ty),+) => {$(
        impl<'a, T, V> Mul<Col<V>, Col<Box<[T]>>> for $ty where
            T: Gemv + One + Zero,
            V: Vector<T>,
        {
            fn mul(&self, rhs: &Col<V>) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, T, V> Mul<Col<V>, Col<Box<[T]>>> for Trans<$ty> where
            T: Gemv + One + Zero,
            V: Vector<T>,
        {
            fn mul(&self, rhs: &Col<V>) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, T, M> Mul<M, Mat<T>> for $ty where T: Gemm + One + Zero, M: blas::Matrix<T> {
            fn mul(&self, rhs: &M) -> Mat<T> {
                mm(self, rhs)
            }
        }

        impl<'a, T, M> Mul<M, Mat<T>> for Trans<$ty> where
            T: Gemm + One + Zero,
            M: blas::Matrix<T>,
        {
            fn mul(&self, rhs: &M) -> Mat<T> {
                mm(self, rhs)
            }
        })+
    }
}

view!(View<'a, T>, MutView<'a, T>)

fn mc<T, M, V>(lhs: &M, rhs: &Col<V>) -> Col<Box<[T]>> where
    T: Gemv + One + Zero,
    M: blas::Matrix<T>,
    V: Vector<T>,
{
    assert_eq!(lhs.ncols(), rhs.nrows());
    assert!(lhs.ncols() != 0);

    if lhs.nrows() == 0 { return Col::new(box []) }

    let gemv = Gemv::gemv(None::<T>);
    let trans = lhs.trans();
    let (m, n) = match trans {
        Transpose::No => (lhs.nrows().to_blasint(), lhs.ncols().to_blasint()),
        Transpose::Yes => (lhs.ncols().to_blasint(), lhs.nrows().to_blasint()),
    };
    let alpha = One::one();
    let a = lhs.as_ptr();
    let lda = lhs.stride().unwrap_or(m);
    let x = rhs.0.as_ptr();
    let incx = rhs.0.stride();
    let beta = Zero::zero();
    let mut data = Vec::with_capacity(lhs.nrows());
    let y = data.as_mut_ptr();
    let incy = One::one();

    unsafe {
        gemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
        data.set_len(lhs.nrows());
    }

    Col::new(data.into_boxed_slice())
}

fn mm<T, L, R>(lhs: &L, rhs: &R) -> Mat<T> where
    T: Gemm + One + Zero,
    L: blas::Matrix<T>,
    R: blas::Matrix<T>,
{
    assert_eq!(lhs.ncols(), rhs.nrows());
    assert!(lhs.ncols() != 0);

    if lhs.nrows() == 0 || rhs.ncols() == 0 {
        return unsafe { Mat::from_parts(box [], (lhs.nrows(), rhs.ncols())) }
    }

    let length = lhs.nrows().checked_mul(rhs.ncols()).unwrap();

    let gemm = Gemm::gemm(None::<T>);
    let transa = lhs.trans();
    let transb = rhs.trans();
    let m = lhs.nrows().to_blasint();
    let k = lhs.ncols().to_blasint();
    let n = rhs.ncols().to_blasint();
    let alpha = One::one();
    let a = lhs.as_ptr();
    let lda = lhs.stride().unwrap_or_else(|| match transa {
        Transpose::No => m,
        Transpose::Yes => k,
    });
    let b = rhs.as_ptr();
    let ldb = rhs.stride().unwrap_or_else(|| match transb {
        Transpose::No => k,
        Transpose::Yes => n,
    });
    let beta = Zero::zero();
    let mut data = Vec::with_capacity(length);
    let c = data.as_mut_ptr();
    let ldc = m;

    unsafe {
        gemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
        data.set_len(length);
        Mat::from_parts(data.into_boxed_slice(), (lhs.nrows(), rhs.ncols()))
    }
}

fn rm<T, M, V>(lhs: &Row<V>, rhs: &M) -> Row<Box<[T]>> where
    T: Gemv + One + Zero,
    M: blas::Matrix<T>,
    V: Vector<T>,
{
    assert_eq!(lhs.ncols(), rhs.nrows());
    assert!(lhs.ncols() != 0);

    if rhs.ncols() == 0 { return Row::new(box []) }

    let gemv = Gemv::gemv(None::<T>);
    let trans = !rhs.trans();
    let (m, n) = match trans {
        Transpose::No => (rhs.ncols().to_blasint(), rhs.nrows().to_blasint()),
        Transpose::Yes => (rhs.nrows().to_blasint(), rhs.ncols().to_blasint()),
    };
    let alpha = One::one();
    let a = rhs.as_ptr();
    let lda = rhs.stride().unwrap_or(m);
    let x = lhs.0.as_ptr();
    let incx = lhs.0.stride();
    let beta = Zero::zero();
    let mut data = Vec::with_capacity(rhs.ncols());
    let y = data.as_mut_ptr();
    let incy = One::one();

    unsafe {
        gemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
        data.set_len(rhs.ncols());
    }

    Row::new(data.into_boxed_slice())
}
