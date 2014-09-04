use blas::{blasint, ffi, BlasAccelerated};

type Signature<T> = unsafe extern "C" fn (
    *const i8,
    *const i8,
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

pub trait BlasGemm: BlasAccelerated {
    fn gemm(Option<Self>) -> Signature<Self>;
}

impl BlasGemm for f32 {
    fn gemm(_: Option<f32>) -> Signature<f32> {
        ffi::sgemm_
    }
}

impl BlasGemm for f64 {
    fn gemm(_: Option<f64>) -> Signature<f64> {
        ffi::dgemm_
    }
}
