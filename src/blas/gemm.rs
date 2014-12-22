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
// FIXME (UFCS) Get rid of `Option<Self>`
pub trait Gemm {
    /// Returns the foreign `gemm` function
    fn gemm(Option<Self>) -> Fn<Self>;
}

impl Gemm for Complex<f32> {
    fn gemm(_: Option<Complex<f32>>) -> Fn<Complex<f32>> {
        ffi::cgemm_
    }
}

impl Gemm for Complex<f64> {
    fn gemm(_: Option<Complex<f64>>) -> Fn<Complex<f64>> {
        ffi::zgemm_
    }
}

impl Gemm for f32 {
    fn gemm(_: Option<f32>) -> Fn<f32> {
        ffi::sgemm_
    }
}

impl Gemm for f64 {
    fn gemm(_: Option<f64>) -> Fn<f64> {
        ffi::dgemm_
    }
}
