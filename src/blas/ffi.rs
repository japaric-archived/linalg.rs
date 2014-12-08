//! FFI

use blas::Transpose;
use libc::{c_double, c_float};

use complex::Complex;

/// The integer used by the BLAS library
// TODO Handle 64-bit BLAS
#[allow(non_camel_case_types)]
pub type blasint = ::libc::c_int;

// y := alpha * x + y
#[link(name="blas")]
extern {
    pub fn caxpy_(
        n: *const blasint,
        alpha: *const Complex<c_float>,
        x: *const Complex<c_float>,
        incx: *const blasint,
        y: *mut Complex<c_float>,
        incy: *const blasint,
    );

    pub fn daxpy_(
        n: *const blasint,
        alpha: *const c_double,
        x: *const c_double,
        incx: *const blasint,
        y: *mut c_double,
        incy: *const blasint,
    );

    pub fn saxpy_(
        n: *const blasint,
        alpha: *const c_float,
        x: *const c_float,
        incx: *const blasint,
        y: *mut c_float,
        incy: *const blasint,
    );

    pub fn zaxpy_(
        n: *const blasint,
        alpha: *const Complex<c_double>,
        x: *const Complex<c_double>,
        incx: *const blasint,
        y: *mut Complex<c_double>,
        incy: *const blasint,
    );
}

// y := x
#[link(name="blas")]
extern {
    pub fn ccopy_(
        n: *const blasint,
        x: *const Complex<c_float>,
        incx: *const blasint,
        y: *mut Complex<c_float>,
        incy: *const blasint,
    );

    pub fn dcopy_(
        n: *const blasint,
        x: *const c_double,
        incx: *const blasint,
        y: *mut c_double,
        incy: *const blasint,
    );

    pub fn scopy_(
        n: *const blasint,
        x: *const c_float,
        incx: *const blasint,
        y: *mut c_float,
        incy: *const blasint,
    );

    pub fn zcopy_(
        n: *const blasint,
        x: *const Complex<c_double>,
        incx: *const blasint,
        y: *mut Complex<c_double>,
        incy: *const blasint,
    );
}

// dot := x^T * y
#[link(name="blas")]
extern {
    pub fn ddot_(
        n: *const blasint,
        x: *const c_double,
        incx: *const blasint,
        y: *const c_double,
        incy: *const blasint,
    ) -> c_double;

    pub fn sdot_(
        n: *const blasint,
        x: *const c_float,
        incx: *const blasint,
        y: *const c_float,
        incy: *const blasint,
    ) -> c_float;
}

// y := alpha * A * x + beta * y
#[link(name="blas")]
extern {
    pub fn cgemv_(
        trans: *const Transpose,
        m: *const blasint,
        n: *const blasint,
        alpha: *const Complex<c_float>,
        a: *const Complex<c_float>,
        lda: *const blasint,
        x: *const Complex<c_float>,
        incx: *const blasint,
        beta: *const Complex<c_float>,
        y: *mut Complex<c_float>,
        incy: *const blasint,
    );

    pub fn dgemv_(
        trans: *const Transpose,
        m: *const blasint,
        n: *const blasint,
        alpha: *const c_double,
        a: *const c_double,
        lda: *const blasint,
        x: *const c_double,
        incx: *const blasint,
        beta: *const c_double,
        y: *mut c_double,
        incy: *const blasint,
    );

    pub fn sgemv_(
        trans: *const Transpose,
        m: *const blasint,
        n: *const blasint,
        alpha: *const c_float,
        a: *const c_float,
        lda: *const blasint,
        x: *const c_float,
        incx: *const blasint,
        beta: *const c_float,
        y: *mut c_float,
        incy: *const blasint,
    );

    pub fn zgemv_(
        trans: *const Transpose,
        m: *const blasint,
        n: *const blasint,
        alpha: *const Complex<c_double>,
        a: *const Complex<c_double>,
        lda: *const blasint,
        x: *const Complex<c_double>,
        incx: *const blasint,
        beta: *const Complex<c_double>,
        y: *mut Complex<c_double>,
        incy: *const blasint,
    );
}

// C := alpha * A * B + beta * C
#[link(name="blas")]
extern {
    pub fn cgemm_(
        transa: *const Transpose,
        transb: *const Transpose,
        m: *const blasint,
        n: *const blasint,
        k: *const blasint,
        alpha: *const Complex<c_float>,
        a: *const Complex<c_float>,
        lda: *const blasint,
        b: *const Complex<c_float>,
        ldb: *const blasint,
        beta: *const Complex<c_float>,
        c: *mut Complex<c_float>,
        ldc: *const blasint,
    );

    pub fn dgemm_(
        transa: *const Transpose,
        transb: *const Transpose,
        m: *const blasint,
        n: *const blasint,
        k: *const blasint,
        alpha: *const c_double,
        a: *const c_double,
        lda: *const blasint,
        b: *const c_double,
        ldb: *const blasint,
        beta: *const c_double,
        c: *mut c_double,
        ldc: *const blasint,
    );

    pub fn sgemm_(
        transa: *const Transpose,
        transb: *const Transpose,
        m: *const blasint,
        n: *const blasint,
        k: *const blasint,
        alpha: *const c_float,
        a: *const c_float,
        lda: *const blasint,
        b: *const c_float,
        ldb: *const blasint,
        beta: *const c_float,
        c: *mut c_float,
        ldc: *const blasint,
    );

    pub fn zgemm_(
        transa: *const Transpose,
        transb: *const Transpose,
        m: *const blasint,
        n: *const blasint,
        k: *const blasint,
        alpha: *const Complex<c_double>,
        a: *const Complex<c_double>,
        lda: *const blasint,
        b: *const Complex<c_double>,
        ldb: *const blasint,
        beta: *const Complex<c_double>,
        c: *mut Complex<c_double>,
        ldc: *const blasint,
    );
}
