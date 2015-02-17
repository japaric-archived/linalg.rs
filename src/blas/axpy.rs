//! y := alpha * x + y

use complex::Complex;

use blas::{blasint, ffi};

/// The signature of `axpy`
pub type Fn<T> = unsafe extern "C" fn (
    *const blasint,
    *const T,
    *const T,
    *const blasint,
    *mut T,
    *const blasint,
);

/// Types with `axpy` acceleration
pub trait Axpy {
    /// Returns the foreign `axpy` function
    fn axpy() -> Fn<Self>;
}

impl Axpy for Complex<f32> {
    fn axpy() -> Fn<Complex<f32>> {
        ffi::caxpy_
    }
}

impl Axpy for Complex<f64> {
    fn axpy() -> Fn<Complex<f64>> {
        ffi::zaxpy_
    }
}

impl Axpy for f32 {
    fn axpy() -> Fn<f32> {
        ffi::saxpy_
    }
}

impl Axpy for f64 {
    fn axpy() -> Fn<f64> {
        ffi::daxpy_
    }
}
