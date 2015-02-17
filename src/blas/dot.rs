//! dot := x^T * y

use blas::{blasint, ffi};

/// The signature of `dot`
pub type Fn<T> = unsafe extern "C" fn (
    *const blasint,
    *const T,
    *const blasint,
    *const T,
    *const blasint,
) -> T;

/// Types with `dot` acceleration
pub trait Dot {
    /// Returns the foreign `dot` function
    fn dot() -> Fn<Self>;
}

impl Dot for f32 {
    fn dot() -> Fn<f32> {
        ffi::sdot_
    }
}

impl Dot for f64 {
    fn dot() -> Fn<f64> {
        ffi::ddot_
    }
}
