[![Build Status](https://travis-ci.org/japaric/linalg.rs.svg?branch=master)](https://travis-ci.org/japaric/linalg.rs)

# What is this?

An **experimental** linear algebra library with
[OpenBLAS acceleration](http://www.openblas.net) written in
[Rust](http://www.rust-lang.org).

# Here be dragons

Very early stage, API may change, shrink, expand, disappear, etc.

There are no docs!

# Done so far

* Vect and Mat
  * Constructors
    * from_elem
    * from_fn
    * ones
    * rand
    * zeros
  * Element-wise ops
    * add_assign
      * BLAS accelerated: `f32`, `f64`, `Cmplx<f32>`, `Cmplx<f64>`
    * mul_assign
      * SIMD accelerated: `f32`, `f64`
    * sub_assign
      * BLAS accelerated: `f32`, `f64`, `Cmplx<f32>`, `Cmplx<f64>`
  * Bulk ops
    * all
    * any
    * iter
    * norm2
      * BLAS accelerated: `f32`, `f64`, `Cmplx<f32>`, `Cmplx<f64>`
    * scale
      * BLAS accelerated: `f32`, `f64`, `Cmplx<f32>`, `Cmplx<f64>`

* BLAS
  * FFI
    * Level 1

# Source of inspiration

* [numpy](http://www.numpy.org) and [scipy](http://www.scipy.org) APIs
* [RustAlgebloat](https://github.com/SiegeLord/RustAlgebloat), for its crazy
  templating!
