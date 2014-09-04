use blas::{blasint, ffi, BlasAccelerated};

type Signature<T> = unsafe extern "C" fn (
    *const blasint,
    *const T,
    *const T,
    *const blasint,
    *mut T,
    *const blasint,
);

pub trait BlasAxpy: BlasAccelerated {
    fn axpy(Option<Self>) -> Signature<Self>;
}

impl BlasAxpy for f32 {
    fn axpy(_: Option<f32>) -> Signature<f32> {
        ffi::saxpy_
    }
}

impl BlasAxpy for f64 {
    fn axpy(_: Option<f64>) -> Signature<f64> {
        ffi::daxpy_
    }
}
