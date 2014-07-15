use num::Complex;
use std::fmt::Show;
use std::num;
use std::simd::{f32x4,f64x2};
use std::slice::Items;

use blas::ffi;
use self::traits::{ArrayNorm2,ArrayScale,ArrayShape};
// FIXME mozilla/rust#5992 Use std {Add,Mul,Sub}Assign
use traits::{AddAssign,Iterable,MulAssign,SubAssign};

pub mod traits;

#[deriving(Clone, PartialEq, Show, Eq)]
pub struct Array<S, T> {
    data: Vec<T>,
    shape: S,
}

impl<
    S,
    T
> Array<S, T> {
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        self.data.as_slice()
    }

    #[inline]
    pub fn slice<'a>(&'a self, start: uint, end: uint) -> &'a [T] {
        self.data.slice(start, end)
    }

    // TODO fork-join parallelism?
    #[inline]
    pub fn map(&mut self, op: |&T| -> T) {
        for x in self.data.mut_iter() {
            *x = op(x);
        }
    }

    #[inline]
    pub fn unwrap(self) -> Vec<T> {
        self.data
    }

    #[inline]
    pub unsafe fn from_raw_parts(v: Vec<T>, s: S) -> Array<S, T> {
        Array {
            data: v,
            shape: s,
        }
    }
}

// AddAssign
// FIXME mozilla/rust#7059 convert to generic fallback
impl<
    S: Clone + Eq + Show
> AddAssign<Array<S, int>>
for Array<S, int> {
    #[inline]
    fn add_assign(&mut self, rhs: &Array<S, int>) {
        assert_shape!(self, rhs, add_assign, +=)

        for (lhs, rhs) in self.data.mut_iter().zip(rhs.iter()) {
            *lhs = *lhs + *rhs;
        }
    }
}

macro_rules! add_assign {
    ($ty:ty, $ffi:ident) => {
        impl<
            S: Clone + Eq + Show
        > AddAssign<Array<S, $ty>>
        for Array<S, $ty> {
            #[inline]
            fn add_assign(&mut self, rhs: &Array<S, $ty>) {
                assert_shape!(self, rhs, add_assign, +=)

                let plus_one = num::one::<$ty>();

                unsafe {
                    ffi::$ffi(&(self.len() as int), &plus_one,
                              rhs.as_ptr(), &1,
                              self.as_mut_ptr(), &1)
                }
            }
        }
    }
}

add_assign!(f32, saxpy_)
add_assign!(f64, daxpy_)
add_assign!(Complex<f32>, caxpy_)
add_assign!(Complex<f64>, zaxpy_)

// ArrayNorm2
macro_rules! norm2 {
    ($inp:ty, $out:ty, $ffi:ident) => {
        impl<
            S
        > ArrayNorm2<$out>
        for Array<S, $inp> {
            #[inline]
            fn norm2(&self) -> $out {
                unsafe {
                    ffi::$ffi(&(self.len() as int),
                              self.as_ptr(), &1)
                }
            }
        }
    }
}

norm2!(f32, f32, snrm2_)
norm2!(f64, f64, dnrm2_)
norm2!(Complex<f32>, f32, scnrm2_)
norm2!(Complex<f64>, f64, dznrm2_)

// ArrayScale
// FIXME mozilla/rust#7059 convert to generic fallback
impl<
    S
> ArrayScale<int>
for Array<S, int> {
    #[inline]
    fn scale(&mut self, alpha: int) {
        for x in self.data.mut_iter() {
            *x = *x * alpha;
        }
    }
}

macro_rules! scale {
    ($ty:ty, $ffi:ident) => {
        impl<
            S
        > ArrayScale<$ty>
        for Array<S, $ty> {
            #[inline]
            fn scale(&mut self, alpha: $ty) {
                unsafe {
                    ffi::$ffi(&(self.len() as int), &alpha,
                              self.as_mut_ptr(), &1)
                }
            }
        }
    }
}

scale!(f32, sscal_)
scale!(f64, dscal_)
scale!(Complex<f32>, cscal_)
scale!(Complex<f64>, zscal_)

// ArrayShape
impl<
    S: Clone,
    T
> ArrayShape<S>
for Array<S, T> {
    #[inline]
    fn shape(&self) -> S {
        self.shape.clone()
    }
}

impl<
    'a,
    S: Clone,
    T
> ArrayShape<S>
for &'a Array<S, T> {
    #[inline]
    fn shape(&self) -> S {
        (*self).shape()
    }
}

// Collection
impl<
    S,
    T
> Collection
for Array<S, T> {
    #[inline]
    fn len(&self) -> uint {
        self.data.len()
    }
}

// Iterable
impl<
    'a,
    S,
    T
> Iterable<'a, T, Items<'a, T>>
for Array<S, T> {
    #[inline]
    fn iter(&'a self) -> Items<'a, T> {
        self.data.iter()
    }
}

// MulAssign
// TODO specialized MulAssign impl for Complex<f32> and Complex<f64>

// FIXME mozilla/rust#7059 convert to generic fallback
impl<
    S: Clone + Eq + Show
> MulAssign<Array<S, int>>
for Array<S, int> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Array<S, int>) {
        assert_shape!(self, rhs, mul_assign, *=)

        for (lhs, rhs) in self.data.mut_iter().zip(rhs.iter()) {
            *lhs = *lhs * *rhs;
        }
    }
}

// TODO fork-join parallelism?
macro_rules! mul_assign {
    ($ty:ty, $stride:expr, $simd:ty) => {
        impl<
            S: Clone + Eq + Show
        > MulAssign<Array<S, $ty>>
        for Array<S, $ty> {
            #[inline]
            fn mul_assign(&mut self, rhs: &Array<S, $ty>) {
                assert_shape!(self, rhs, mul_assign, *=)

                let n = self.len() as int / $stride;
                let p_self = self.as_mut_ptr();
                let p_rhs = rhs.as_ptr();
                let simd_p_self = p_self as *mut $simd;
                let simd_p_rhs = p_rhs as *const $simd;

                for i in range(0, n) {
                    unsafe {
                        *simd_p_self.offset(i) *= *simd_p_rhs.offset(i);
                    }
                }

                for i in range($stride * n, self.len() as int) {
                    unsafe {
                        *p_self.offset(i) *= *p_rhs.offset(i);
                    }
                }
            }
        }
    }
}

mul_assign!(f32, 4, f32x4)
mul_assign!(f64, 2, f64x2)

// SubAssign
// FIXME mozilla/rust#7059 convert to generic fallback
impl<
    S: Clone + Eq + Show
> SubAssign<Array<S, int>>
for Array<S, int> {
    #[inline]
    fn sub_assign(&mut self, rhs: &Array<S, int>) {
        assert_shape!(self, rhs, sub_assign, -=)

        for (lhs, rhs) in self.data.mut_iter().zip(rhs.iter()) {
            *lhs = *lhs - *rhs;
        }
    }
}

macro_rules! sub_assign {
    ($ty:ty, $ffi:ident) => {
        impl<
            S: Clone + Eq + Show
        > SubAssign<Array<S, $ty>>
        for Array<S, $ty> {
            #[inline]
            fn sub_assign(&mut self, rhs: &Array<S, $ty>) {
                assert_shape!(self, rhs, sub_assign, -=)

                let minus_one = -num::one::<$ty>();

                unsafe {
                    ffi::$ffi(&(self.len() as int), &minus_one,
                              rhs.as_ptr(), &1,
                              self.as_mut_ptr(), &1)
                }
            }
        }
    }
}

sub_assign!(f32, saxpy_)
sub_assign!(f64, daxpy_)
sub_assign!(Complex<f32>, caxpy_)
sub_assign!(Complex<f64>, zaxpy_)
