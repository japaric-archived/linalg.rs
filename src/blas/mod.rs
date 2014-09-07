use num::Complex;
use std::num;

mod ffi;

pub mod axpy;
pub mod copy;
pub mod dot;
pub mod gemm;

// TODO Handle 64-bit BLAS
#[allow(non_camel_case_types)]
pub type blasint = ::libc::c_int;

pub static BLAS_NO_TRANS: i8 = 'n' as i8;

pub fn to_blasint<N: NumCast>(n: N) -> blasint {
    num::cast(n).expect("casting to blasint failed")
}

pub trait BlasAccelerated {}

impl BlasAccelerated for Complex<f32> {}
impl BlasAccelerated for Complex<f64> {}
impl BlasAccelerated for f32 {}
impl BlasAccelerated for f64 {}

pub trait BlasMutPtr<T> {
    fn blas_mut_ptr(&mut self) -> *mut T;
}

macro_rules! blas_mut_ptr {
    () => {
        fn blas_mut_ptr(&mut self) -> *mut T {
            self.as_mut_ptr()
        }
    }
}

impl<'a, T: BlasAccelerated> BlasMutPtr<T> for &'a mut [T] { blas_mut_ptr!() }
impl<T: BlasAccelerated> BlasMutPtr<T> for Vec<T> { blas_mut_ptr!() }

pub trait BlasPtr<T> {
    fn blas_ptr(&self) -> *const T;
}

macro_rules! blas_ptr {
    () => {
        fn blas_ptr(&self) -> *const T {
            self.as_ptr()
        }
    }
}

impl<'a, T: BlasAccelerated> BlasPtr<T> for &'a [T] { blas_ptr!() }
impl<'a, T: BlasAccelerated> BlasPtr<T> for &'a mut [T] { blas_ptr!() }
impl<T: BlasAccelerated> BlasPtr<T> for Vec<T> { blas_ptr!() }

pub trait BlasStride {
    fn blas_stride(&self) -> blasint {
        1
    }
}

impl<T: BlasAccelerated> BlasStride for Vec<T> {}
impl<'a, T: BlasAccelerated> BlasStride for &'a [T] {}
impl<'a, T: BlasAccelerated> BlasStride for &'a mut [T] {}
