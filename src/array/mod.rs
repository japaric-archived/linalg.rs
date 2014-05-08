use blas::ffi;
use mat::Mat;
use num::complex::Cmplx;
use self::traits::{ArrayDot,ArrayNorm2,ArrayScale};
use std::fmt::Show;
use std::iter::AdditiveIterator;
use std::num::one;
use std::slice::Items;
use std::unstable::simd::{f32x4,f64x2};
// FIXME mozilla/rust#5992 Use std {Add,Mul,Sub}Assign
// FIXME mozilla/rust#6515 Use std Index
use traits::{AddAssign,Index,Iterable,MulAssign,SubAssign,UnsafeIndex};
use vec::Vect;

pub mod traits;

#[deriving(Eq, Show)]
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
    pub fn as_ptr(&self) -> *T {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        self.data.as_slice()
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

impl<
    S: Clone,
    T
> Array<S, T> {
    #[inline]
    pub fn shape(&self) -> S {
        self.shape.clone()
    }
}

macro_rules! assert_shape {
    ($method:ident, $op:tt) => ({
        assert!(self.shape() == rhs.shape(),
                "{}: dimension mismatch: {} {} {}",
                stringify!($method),
                self.shape(),
                stringify!($op),
                rhs.shape());
    })
}

// FIXME mozilla/rust#7059 convert to generic fallback
impl<
    S: Clone + Eq + Show
> AddAssign<Array<S, int>>
for Array<S, int> {
    #[inline]
    fn add_assign(&mut self, rhs: &Array<S, int>) {
        assert_shape!(add_assign, +=)

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
                assert_shape!(add_assign, +=)

                let plus_one = one::<$ty>();

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
add_assign!(Cmplx<f32>, caxpy_)
add_assign!(Cmplx<f64>, zaxpy_)

// ArrayDot: Vect . Vect
// FIXME mozilla/rust#7059 convert to generic fallback
impl
ArrayDot<Vect<int>, int>
for Vect<int> {
    fn dot(&self, rhs: &Vect<int>) -> int {
        assert_shape!(dot, .)

        self.iter().zip(rhs.iter()).map(|(lhs, rhs)| lhs.mul(rhs)).sum()
    }
}

macro_rules! vector_dot {
    ($ty:ty, $ffi:ident) => {
        impl
        ArrayDot<Vect<$ty>, $ty>
        for Vect<$ty> {
            fn dot(&self, rhs: &Vect<$ty>) -> $ty {
                assert_shape!(dot, .)

                unsafe {
                    ffi::$ffi(&(self.len() as int),
                              self.as_ptr(), &1,
                              rhs.as_ptr(), &1)
                }
            }
        }
    }
}

vector_dot!(f32, sdot_)
vector_dot!(f64, ddot_)

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
norm2!(Cmplx<f32>, f32, scnrm2_)
norm2!(Cmplx<f64>, f64, dznrm2_)

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
scale!(Cmplx<f32>, cscal_)
scale!(Cmplx<f64>, zscal_)

impl<
    S,
    T
> Container
for Array<S, T> {
    #[inline]
    fn len(&self) -> uint {
        self.data.len()
    }
}

impl<
    T
> Index<(uint, uint), T>
for Mat<T> {
    #[inline]
    fn index<'a>(&'a self, index: &(uint, uint)) -> &'a T {
        let &(row, col) = index;
        let (nrows, ncols) = self.shape();

        assert!(row < nrows && col < ncols,
                "index: out of bounds: {} of {}", index, self.shape());

        unsafe { self.data.as_slice().unsafe_ref(row * ncols + col) }
    }
}

impl<
    T
> UnsafeIndex<(uint, uint), T>
for Mat<T> {
    #[inline]
    unsafe fn unsafe_index<'a>(&'a self, index: &(uint, uint)) -> &'a T {
        let &(row, col) = index;
        let (_, ncols) = self.shape();

        self.data.as_slice().unsafe_ref(row * ncols + col)
    }
}

impl<
    T
> Index<uint, T>
for Vect<T> {
    #[inline]
    fn index<'a>(&'a self, index: &uint) -> &'a T {
        assert!(*index < self.len(),
                "index: out of bounds: {} of {}", index, self.len());

        unsafe { self.unsafe_index(index) }
    }
}

impl<
    T
> UnsafeIndex<uint, T>
for Vect<T> {
    #[inline]
    unsafe fn unsafe_index<'a>(&'a self, index: &uint) -> &'a T {
        self.data.as_slice().unsafe_ref(*index)
    }
}

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

// FIXME mozilla/rust#7059 convert to generic fallback
impl<
    S: Clone + Eq + Show
> MulAssign<Array<S, int>>
for Array<S, int> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Array<S, int>) {
        assert_shape!(mul_assign, *=)

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
                assert_shape!(mul_assign, *=)

                let n = self.len() as int / $stride;
                let p_self = self.as_mut_ptr();
                let p_rhs = rhs.as_ptr();
                let simd_p_self = p_self as *mut $simd;
                let simd_p_rhs = p_rhs as *$simd;

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

// TODO specialized MulAssign impl for Cmplx<f32> and Cmplx<f64>

// FIXME mozilla/rust#7059 convert to generic fallback
impl<
    S: Clone + Eq + Show
> SubAssign<Array<S, int>>
for Array<S, int> {
    #[inline]
    fn sub_assign(&mut self, rhs: &Array<S, int>) {
        assert_shape!(sub_assign, -=)

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
                assert_shape!(sub_assign, -=)

                let minus_one = -one::<$ty>();

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
sub_assign!(Cmplx<f32>, caxpy_)
sub_assign!(Cmplx<f64>, zaxpy_)
