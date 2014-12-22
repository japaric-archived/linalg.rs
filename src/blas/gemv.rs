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
// FIXME (UFCS) Get rid of `Option<Self>`
pub trait Gemv {
    /// Returns the foreign `gemv` function
    fn gemv(Option<Self>) -> Fn<Self>;
}

impl Gemv for Complex<f32> {
    fn gemv(_: Option<Complex<f32>>) -> Fn<Complex<f32>> {
        ffi::cgemv_
    }
}

impl Gemv for Complex<f64> {
    fn gemv(_: Option<Complex<f64>>) -> Fn<Complex<f64>> {
        ffi::zgemv_
    }
}

impl Gemv for f32 {
    fn gemv(_: Option<f32>) -> Fn<f32> {
        ffi::sgemv_
    }
}

impl Gemv for f64 {
    fn gemv(_: Option<f64>) -> Fn<f64> {
        ffi::dgemv_
    }
}
