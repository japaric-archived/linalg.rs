//! y := alpha * A * x + beta * y

use blas::{Transpose, blasint, ffi};
use complex::Complex;

/// The signature of `gemv`
pub type Fn<T> = unsafe extern "C" fn (
    *const Transpose,
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


/// Types with `gemv` acceleration
pub trait Gemv {
    /// Returns the foreign `gemv` function
    fn gemv() -> Fn<Self>;
}

impl Gemv for Complex<f32> {
    fn gemv() -> Fn<Complex<f32>> {
        ffi::cgemv_
    }
}

impl Gemv for Complex<f64> {
    fn gemv() -> Fn<Complex<f64>> {
        ffi::zgemv_
    }
}

impl Gemv for f32 {
    fn gemv() -> Fn<f32> {
        ffi::sgemv_
    }
}

impl Gemv for f64 {
    fn gemv() -> Fn<f64> {
        ffi::dgemv_
    }
}
