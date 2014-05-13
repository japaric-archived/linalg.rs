#![allow(ctypes)]

use num::complex::Complex;

#[link(name = "blas")]
extern {
    // y <- alpha * X + Y
    pub fn saxpy_(N: *int, alpha: *f32,
                  x: *f32, inc_x: *int,
                  y: *mut f32, inc_y: *int);
    pub fn daxpy_(N: *int, alpha: *f64,
                  x: *f64, inc_x: *int,
                  y: *mut f64, inc_y: *int);
    pub fn caxpy_(N: *int, alpha: *Complex<f32>,
                  x: *Complex<f32>, inc_x: *int,
                  y: *mut Complex<f32>, inc_y: *int);
    pub fn zaxpy_(N: *int, alpha: *Complex<f64>,
                  x: *Complex<f64>, inc_x: *int,
                  y: *mut Complex<f64>, inc_y: *int);

    // Y <- X
    pub fn scopy_(N: *int,
                  x: *f32, inc_x: *int,
                  y: *mut f32, inc_y: *int);
    pub fn dcopy_(N: *int,
                  x: *f64, inc_x: *int,
                  y: *mut f64, inc_y: *int);
    pub fn ccopy_(N: *int,
                  x: *Complex<f32>, inc_x: *int,
                  y: *mut Complex<f32>, inc_y: *int);
    pub fn zcopy_(N: *int,
                  x: *Complex<f64>, inc_x: *int,
                  y: *mut Complex<f64>, inc_y: *int);

    // dot <- X^T * Y
    pub fn sdot_(N: *int,
                 x: *f32, inc_x: *int,
                 y: *f32, inc_y: *int) -> f32;
    pub fn ddot_(N: *int,
                 x: *f64, inc_x: *int,
                 y: *f64, inc_y: *int) -> f64;

    // nrm2 <- ||X||_2
    pub fn snrm2_(N: *int,
                 x: *f32, inc_x: *int) -> f32;
    pub fn dnrm2_(N: *int,
                 x: *f64, inc_x: *int) -> f64;
    pub fn scnrm2_(N: *int,
                   x: *Complex<f32>, inc_x: *int) -> f32;
    pub fn dznrm2_(N: *int,
                   x: *Complex<f64>, inc_x: *int) -> f64;

    // X <- alpha * X
    pub fn sscal_(N: *int, alpha: *f32,
                  x: *mut f32, inc_x: *int);
    pub fn dscal_(N: *int, alpha: *f64,
                  x: *mut f64, inc_x: *int);
    pub fn cscal_(N: *int, alpha: *Complex<f32>,
                  x: *mut Complex<f32>, inc_x: *int);
    pub fn zscal_(N: *int, alpha: *Complex<f64>,
                  x: *mut Complex<f64>, inc_x: *int);

    // Y <-> X
    pub fn sswap_(N: *int,
                  x: *mut f32, inc_x: *int,
                  y: *mut f32, inc_y: *int);
    pub fn dswap_(N: *int,
                  x: *mut f64, inc_x: *int,
                  y: *mut f64, inc_y: *int);
    pub fn cswap_(N: *int,
                  x: *mut Complex<f32>, inc_x: *int,
                  y: *mut Complex<f32>, inc_y: *int);
    pub fn zswap_(N: *int,
                  x: *mut Complex<f64>, inc_x: *int,
                  y: *mut Complex<f64>, inc_y: *int);
}
