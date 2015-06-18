//! BLAS

use blas::{Axpy, Dot, Gemm, Gemv, Nrm2};

use ops::Mat;
use strided::{Col, Slice};
use traits::{Matrix, Transpose};

/// y <- alpha * x + y
pub fn axpy<T>(alpha: &T, x: &Slice<T>, y: &mut Slice<T>) where
    T: Axpy,
{
    unsafe {
        assert_eq!(x.len(), y.len());

        let axpy = T::axpy();

        let ::strided::raw::Slice { data, len, stride } = x.repr();
        let ref n = len.i32();
        let x = *data;
        let ref incx = stride.i32();

        let ::strided::raw::Slice { data, stride, .. } = y.repr();
        let y = *data;
        let ref incy = stride.i32();

        debug!("axpy(n={}, x={:?}), incx={}, y={:?}, incy={}", n, x, incx, y, incy);

        axpy(n, alpha, x, incx, y, incy)
    }
}

/// dot <- x * y
pub fn dot<T>(x: &::strided::Row<T>, y: &::strided::Col<T>) -> T where T: Dot {
    unsafe {
        assert_eq!(x.len(), y.len());

        let dot = T::dot();

        let ::strided::raw::Slice { data, len, stride } = x.repr();
        let ref n = len.i32();
        let x = *data;
        let ref incx = stride.i32();

        let ::strided::raw::Slice { data, stride, .. } = y.repr();
        let y = *data;
        let ref incy = stride.i32();

        debug!("dot(n={}, x={:?}, incx={}, y={:?}), incy={}", n, x, incx, y, incy);

        dot(n, x, incx, y, incy)
    }
}

/// C <- alpha * A * B + beta * C
// NOTE Core
pub fn gemm<T>(alpha: &T, a: &Mat<T>, b: &Mat<T>, beta: &T, c: &mut Mat<T>) where
    T: Gemm,
{
    unsafe {
        assert!(a.ncols() != 0);
        assert_eq!(a.ncols(), b.nrows());
        assert_eq!(c.size(), (a.nrows(), b.ncols()));

        let gemm = T::gemm();

        let ::ops::raw::Mat { data, stride, order, .. } = c.repr();
        let c = *data;
        let ref ldc = stride.i32();
        // Fortran uses column major order, if C is in row major order then we have to transpose
        // the operation: `C' <- (alpha * A * B + beta * C)'`
        let (a, b) = match order {
            ::Order::Col => (a, b),
            // Recall that `(A * B)' = B' * A'`
            ::Order::Row => (b.t(), a.t()),
        };

        let ::ops::raw::Mat { data, nrows, ncols, stride, order, .. } = a.repr();
        let ref transa = order.trans();
        let ref m = nrows.i32();
        let ref k = ncols.i32();
        let a = *data;
        let ref lda = stride.i32();

        let ::ops::raw::Mat { data, ncols, stride, order, .. } = b.repr();
        let ref n = ncols.i32();
        let ref transb = order.trans();
        let b = *data;
        let ref ldb = stride.i32();

        debug!("gemm(transa={:?}, transb={:?}, m={}, n={}, k={}, a={:?}, lda={}, b={:?}, ldb={}, \
            c={:?}, ldc={})", transa, transb, m, n, k, a, lda, b, ldb, c, ldc);

        gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
}

/// y <- alpha * A * x + beta * y
// NOTE Core
pub fn gemv<T>(alpha: &T, a: &Mat<T>, x: &Col<T>, beta: &T, y: &mut Col<T>) where
    T: Gemv,
{
    unsafe {
        assert_eq!(a.ncols(), x.len());
        assert_eq!(a.nrows(), y.len());

        let gemv = T::gemv();

        let ::ops::raw::Mat { data, nrows, ncols, stride, order, .. } = a.repr();
        let ref trans = order.trans();
        let ref m = nrows.i32();
        let ref n = ncols.i32();
        let a = *data;
        let ref lda = stride.i32();

        let ::strided::raw::Slice { data, stride, .. } = x.repr();
        let x = *data;
        let ref incx = stride.i32();

        let ::strided::raw::Slice { data, stride, .. } = y.repr();
        let y = *data;
        let ref incy = stride.i32();

        debug!("gemv(trans={:?}, m={}, n={}, a={:?}, lda={}, x={:?}, incx={}, y={:?}, incy={})",
            trans, m, n, a, lda, x, incx, y, incy);

        gemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
}

/// nrm2 <- ||x||_2
pub fn nrm2<T>(x: &Slice<T>) -> T::Output where T: Nrm2 {
    unsafe {
        let nrm2 = T::nrm2();

        let ::strided::raw::Slice { data, len, stride } = x.repr();
        let ref n = len.i32();
        let x = *data;
        let ref incx = stride.i32();

        nrm2(n, x, incx)
    }
}
