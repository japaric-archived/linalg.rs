//! C := alpha * A * B + beta * C

use blas::{Transpose, blasint, ffi};
use complex::Complex;

/// The signature of `gemm`
pub type Fn<T> = unsafe extern "C" fn (
    *const Transpose,
    *const Transpose,
    *const blasint,
    *const blasint,
    *const blasint,
    *const T,
    *const T,
    *const blasint,
    *const T,
    *const blasint,
    *const T,
    *mut T,
    *const blasint,
);


/// Types with `gemm` acceleration
pub trait Gemm {
    /// Returns the foreign `gemm` function
    fn gemm() -> Fn<Self>;
}

impl Gemm for Complex<f32> {
    fn gemm() -> Fn<Complex<f32>> {
        ffi::cgemm_
    }
}

impl Gemm for Complex<f64> {
    fn gemm() -> Fn<Complex<f64>> {
        ffi::zgemm_
    }
}

impl Gemm for f32 {
    fn gemm() -> Fn<f32> {
        ffi::sgemm_
    }
}

impl Gemm for f64 {
    fn gemm() -> Fn<f64> {
        ffi::dgemm_
    }
}
