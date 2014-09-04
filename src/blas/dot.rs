use blas::{blasint, ffi, BlasAccelerated};

type Signature<T> = unsafe extern "C" fn (
    *const blasint,
    *const T,
    *const blasint,
    *const T,
    *const blasint,
) -> T;

pub trait BlasDot: BlasAccelerated {
    fn dot(Option<Self>) -> Signature<Self>;
}

impl BlasDot for f32 {
    fn dot(_: Option<f32>) -> Signature<f32> {
        ffi::sdot_
    }
}

impl BlasDot for f64 {
    fn dot(_: Option<f64>) -> Signature<f64> {
        ffi::ddot_
    }
}
