//! BLAS

use std::raw::FatPtr;

use blas::{Axpy, Dot, Gemm, Gemv, Nrm2};

use ops::Mat;
use strided::{Col, Vector};
use traits::{Matrix, Transpose};

/// y <- alpha * x + y
pub fn axpy<T>(alpha: &T, x: &Vector<T>, y: &mut Vector<T>) where
    T: Axpy,
{
    assert_eq!(x.len(), y.len());

    let axpy = T::axpy();

    let FatPtr { data, info } = x.repr();
    let ref n = info.len.i32();
    let x = data;
    let ref incx = info.stride.i32();

    let FatPtr { data, info } = y.repr();
    let y = data;
    let ref incy = info.stride.i32();

    info!("axpy(n={}, x={:?}), incx={}, y={:?}, incy={}", n, x, incx, y, incy);

    unsafe {
        axpy(n, alpha, x, incx, y, incy)
    }
}

/// dot <- x * y
pub fn dot<T>(x: &::strided::Row<T>, y: &::strided::Col<T>) -> T where T: Dot {
    assert_eq!(x.len(), y.len());

    let dot = T::dot();

    let FatPtr { data, info } = x.repr();
    let ref n = info.len.i32();
    let x = data;
    let ref incx = info.stride.i32();

    let FatPtr { data, info } = y.repr();
    let y = data;
    let ref incy = info.stride.i32();

    info!("dot(n={}, x={:?}, incx={}, y={:?}), incy={}", n, x, incx, y, incy);

    unsafe {
        dot(n, x, incx, y, incy)
    }
}

/// C <- alpha * A * B + beta * C
// NOTE Core
pub fn gemm<T>(alpha: &T, a: &Mat<T>, b: &Mat<T>, beta: &T, c: &mut Mat<T>) where
    T: Gemm,
{
    assert!(a.ncols() != 0);
    assert_eq!(a.ncols(), b.nrows());
    assert_eq!(c.size(), (a.nrows(), b.ncols()));

    let gemm = T::gemm();

    let FatPtr { data, info } = c.repr();
    let c = data;
    let ref ldc = info.stride.i32();
    // Fortran uses column major order, if C is in row major order then we have to transpose
    // the operation: `C' <- (alpha * A * B + beta * C)'`
    let (a, b) = match info.order {
        ::Order::Col => (a, b),
        // Recall that `(A * B)' = B' * A'`
        ::Order::Row => (b.t(), a.t()),
    };

    let FatPtr { data, info } = a.repr();
    let ref transa = info.order.trans();
    let ref m = info.nrows.i32();
    let ref k = info.ncols.i32();
    let a = data;
    let ref lda = info.stride.i32();

    let FatPtr { data, info } = b.repr();
    let ref n = info.ncols.i32();
    let ref transb = info.order.trans();
    let b = data;
    let ref ldb = info.stride.i32();

    info!("gemm(transa={:?}, transb={:?}, m={}, n={}, k={}, a={:?}, lda={}, b={:?}, ldb={}, \
           c={:?}, ldc={})", transa, transb, m, n, k, a, lda, b, ldb, c, ldc);

    unsafe {
        gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
}

/// y <- alpha * A * x + beta * y
// NOTE Core
pub fn gemv<T>(alpha: &T, a: &Mat<T>, x: &Col<T>, beta: &T, y: &mut Col<T>) where
    T: Gemv,
{
    assert_eq!(a.ncols(), x.len());
    assert_eq!(a.nrows(), y.len());

    let gemv = T::gemv();

    let FatPtr { data, info } = a.repr();
    let ref trans = info.order.trans();
    let ref m = info.nrows.i32();
    let ref n = info.ncols.i32();
    let a = data;
    let ref lda = info.stride.i32();

    let FatPtr { data, info } = x.repr();
    let x = data;
    let ref incx = info.stride.i32();

    let FatPtr { data, info } = y.repr();
    let y = data;
    let ref incy = info.stride.i32();

    info!("gemv(trans={:?}, m={}, n={}, a={:?}, lda={}, x={:?}, incx={}, y={:?}, incy={})",
          trans, m, n, a, lda, x, incx, y, incy);

    unsafe {
        gemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
}

/// nrm2 <- ||x||_2
pub fn nrm2<T>(x: &Vector<T>) -> T::Output where T: Nrm2 {
    let nrm2 = T::nrm2();

    let FatPtr { data, info } = x.repr();
    let ref n = info.len.i32();
    let x = data;
    let ref incx = info.stride.i32();

    info!("nrm2(n={}, x={:?}, incx={})", n, x, incx);

    unsafe {
        nrm2(n, x, incx)
    }
}
