#![allow(ctypes)]

use num::complex::Complex;

#[link(name = "blas")]
extern {
    // y <- alpha * X + Y
    pub fn saxpy_(N: *const int, alpha: *const f32,
                  x: *const f32, inc_x: *const int,
                  y: *mut f32, inc_y: *const int);
    pub fn daxpy_(N: *const int, alpha: *const f64,
                  x: *const f64, inc_x: *const int,
                  y: *mut f64, inc_y: *const int);
    pub fn caxpy_(N: *const int, alpha: *const Complex<f32>,
                  x: *const Complex<f32>, inc_x: *const int,
                  y: *mut Complex<f32>, inc_y: *const int);
    pub fn zaxpy_(N: *const int, alpha: *const Complex<f64>,
                  x: *const Complex<f64>, inc_x: *const int,
                  y: *mut Complex<f64>, inc_y: *const int);

    // Y <- X
    pub fn scopy_(N: *const int,
                  x: *const f32, inc_x: *const int,
                  y: *mut f32, inc_y: *const int);
    pub fn dcopy_(N: *const int,
                  x: *const f64, inc_x: *const int,
                  y: *mut f64, inc_y: *const int);
    pub fn ccopy_(N: *const int,
                  x: *const Complex<f32>, inc_x: *const int,
                  y: *mut Complex<f32>, inc_y: *const int);
    pub fn zcopy_(N: *const int,
                  x: *const Complex<f64>, inc_x: *const int,
                  y: *mut Complex<f64>, inc_y: *const int);

    // dot <- X^T * Y
    pub fn sdot_(N: *const int,
                 x: *const f32, inc_x: *const int,
                 y: *const f32, inc_y: *const int) -> f32;
    pub fn ddot_(N: *const int,
                 x: *const f64, inc_x: *const int,
                 y: *const f64, inc_y: *const int) -> f64;

    // nrm2 <- ||X||_2
    pub fn snrm2_(N: *const int,
                 x: *const f32, inc_x: *const int) -> f32;
    pub fn dnrm2_(N: *const int,
                 x: *const f64, inc_x: *const int) -> f64;
    pub fn scnrm2_(N: *const int,
                   x: *const Complex<f32>, inc_x: *const int) -> f32;
    pub fn dznrm2_(N: *const int,
                   x: *const Complex<f64>, inc_x: *const int) -> f64;

    // X <- alpha * X
    pub fn sscal_(N: *const int, alpha: *const f32,
                  x: *mut f32, inc_x: *const int);
    pub fn dscal_(N: *const int, alpha: *const f64,
                  x: *mut f64, inc_x: *const int);
    pub fn cscal_(N: *const int, alpha: *const Complex<f32>,
                  x: *mut Complex<f32>, inc_x: *const int);
    pub fn zscal_(N: *const int, alpha: *const Complex<f64>,
                  x: *mut Complex<f64>, inc_x: *const int);

    // Y <-> X
    pub fn sswap_(N: *const int,
                  x: *mut f32, inc_x: *const int,
                  y: *mut f32, inc_y: *const int);
    pub fn dswap_(N: *const int,
                  x: *mut f64, inc_x: *const int,
                  y: *mut f64, inc_y: *const int);
    pub fn cswap_(N: *const int,
                  x: *mut Complex<f32>, inc_x: *const int,
                  y: *mut Complex<f32>, inc_y: *const int);
    pub fn zswap_(N: *const int,
                  x: *mut Complex<f64>, inc_x: *const int,
                  y: *mut Complex<f64>, inc_y: *const int);
}
