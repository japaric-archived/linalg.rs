#![allow(ctypes)]

use num::complex::Cmplx;

#[link(name = "blas")]
extern {
    // y <- alpha * X + Y
    pub fn saxpy_(N: *int, alpha: *f32,
                  x: *f32, inc_x: *int,
                  y: *mut f32, inc_y: *int);
    pub fn daxpy_(N: *int, alpha: *f64,
                  x: *f64, inc_x: *int,
                  y: *mut f64, inc_y: *int);
    pub fn caxpy_(N: *int, alpha: *Cmplx<f32>,
                  x: *Cmplx<f32>, inc_x: *int,
                  y: *mut Cmplx<f32>, inc_y: *int);
    pub fn zaxpy_(N: *int, alpha: *Cmplx<f64>,
                  x: *Cmplx<f64>, inc_x: *int,
                  y: *mut Cmplx<f64>, inc_y: *int);

    // Y <- X
    pub fn scopy_(N: *int,
                  x: *f32, inc_x: *int,
                  y: *f32, inc_y: *int);
    pub fn dcopy_(N: *int,
                  x: *f64, inc_x: *int,
                  y: *f64, inc_y: *int);
    pub fn ccopy_(N: *int,
                  x: *Cmplx<f32>, inc_x: *int,
                  y: *Cmplx<f32>, inc_y: *int);
    pub fn zcopy_(N: *int,
                  x: *Cmplx<f64>, inc_x: *int,
                  y: *Cmplx<f64>, inc_y: *int);

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
                   x: *Cmplx<f32>, inc_x: *int) -> f32;
    pub fn dznrm2_(N: *int,
                   x: *Cmplx<f64>, inc_x: *int) -> f64;

    // X <- alpha * X
    pub fn sscal_(N: *int, alpha: *f32,
                  x: *mut f32, inc_x: *int);
    pub fn dscal_(N: *int, alpha: *f64,
                  x: *mut f64, inc_x: *int);
    pub fn cscal_(N: *int, alpha: *Cmplx<f32>,
                  x: *mut Cmplx<f32>, inc_x: *int);
    pub fn zscal_(N: *int, alpha: *Cmplx<f64>,
                  x: *mut Cmplx<f64>, inc_x: *int);

    // Y <-> X
    pub fn sswap_(N: *int,
                  x: *mut f32, inc_x: *int,
                  y: *mut f32, inc_y: *int);
    pub fn dswap_(N: *int,
                  x: *mut f64, inc_x: *int,
                  y: *mut f64, inc_y: *int);
    pub fn cswap_(N: *int,
                  x: *mut Cmplx<f32>, inc_x: *int,
                  y: *mut Cmplx<f32>, inc_y: *int);
    pub fn zswap_(N: *int,
                  x: *mut Cmplx<f64>, inc_x: *int,
                  y: *mut Cmplx<f64>, inc_y: *int);
}
