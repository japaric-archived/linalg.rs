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
