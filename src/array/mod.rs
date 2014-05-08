use blas::ffi;
use num::complex::Cmplx;
use std::fmt::Show;
use std::num::one;
use std::slice::Items;
// FIXME mozilla/rust#5992 Use std {Add,Sub}Assign
use traits::{AddAssign,Iterable,SubAssign};

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
    pub fn as_ptr(&self) -> *T {
        self.data.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline]
    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        self.data.as_slice()
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
