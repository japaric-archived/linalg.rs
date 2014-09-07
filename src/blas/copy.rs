use num::Complex;

use blas::{blasint, ffi, BlasAccelerated};

type Signature<T> = unsafe extern "C" fn (
    *const blasint,
    *const T,
    *const blasint,
    *mut T,
    *const blasint,
);

pub trait BlasCopy: BlasAccelerated {
    fn copy(Option<Self>) -> Signature<Self>;
}

impl BlasCopy for Complex<f32> {
    fn copy(_: Option<Complex<f32>>) -> Signature<Complex<f32>> {
        ffi::ccopy_
    }
}

impl BlasCopy for Complex<f64> {
    fn copy(_: Option<Complex<f64>>) -> Signature<Complex<f64>> {
        ffi::zcopy_
    }
}

impl BlasCopy for f32 {
    fn copy(_: Option<f32>) -> Signature<f32> {
        ffi::scopy_
    }
}

impl BlasCopy for f64 {
    fn copy(_: Option<f64>) -> Signature<f64> {
        ffi::dcopy_
    }
}
