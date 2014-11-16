//! y := x

use blas::ffi::{blasint, mod};
use complex::Complex;

/// The signature of `copy`
pub type Fn<T> = unsafe extern "C" fn (
    *const blasint,
    *const T,
    *const blasint,
    *mut T,
    *const blasint,
);

/// Types with `copy` acceleration
// FIXME (UFCS) Get rid of `Option<Self>`
pub trait Copy {
    /// Returns the foreign `copy` function
    fn copy(Option<Self>) -> Fn<Self>;
}

impl Copy for Complex<f32> {
    fn copy(_: Option<Complex<f32>>) -> Fn<Complex<f32>> {
        ffi::ccopy_
    }
}

impl Copy for Complex<f64> {
    fn copy(_: Option<Complex<f64>>) -> Fn<Complex<f64>> {
        ffi::zcopy_
    }
}

impl Copy for f32 {
    fn copy(_: Option<f32>) -> Fn<f32> {
        ffi::scopy_
    }
}

impl Copy for f64 {
    fn copy(_: Option<f64>) -> Fn<f64> {
        ffi::dcopy_
    }
}
