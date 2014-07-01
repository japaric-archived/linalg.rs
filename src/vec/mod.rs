use rand::Rng;
use rand::distributions::IndependentSample;
use std::iter::AdditiveIterator;
use std::num::{One,Zero};
use std::num;

use array::Array;
use array::traits::{ArrayDot,ArrayShape};
use blas::ffi;
// FIXME mozilla/rust#6515 Use std Index
use traits::{Index,Iterable,UnsafeIndex};

// XXX ugly name, but both Vec and Vector are taken :-(
pub type Vect<T> = Array<(uint,), T>;

#[inline]
pub fn from_elem<T: Clone>(size: uint, elem: T) -> Vect<T> {
    unsafe { Array::from_raw_parts(Vec::from_elem(size, elem), (size,)) }
}

// TODO fork-join parallelism?
#[inline]
pub fn from_fn<T>(size: uint, op: |uint| -> T) -> Vect<T> {
    unsafe { Array::from_raw_parts(Vec::from_fn(size, op), (size,)) }
}

#[inline]
pub fn ones<T: Clone + One>(size: uint) -> Vect<T> {
    from_elem(size, num::one())
}

#[inline]
pub fn rand<
    T,
    D: IndependentSample<T>,
    R: Rng
>(size: uint, dist: &D, rng: &mut R) -> Vect<T> {
    unsafe {
        Array::from_raw_parts(
            range(0, size).map(|_| dist.ind_sample(rng)).collect(),
            (size,)
        )
    }
}

#[inline]
pub fn zeros<T: Clone + Zero>(size: uint) -> Vect<T> {
    from_elem(size, num::zero())
}

// XXX repeated macro, how to DRY?
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

// Index
impl<
    T
> Index<uint, T>
for Vect<T> {
    #[inline]
    fn index<'a>(&'a self, index: &uint) -> &'a T {
        let size = self.len();

        assert!(*index < size, "index: out of bounds: {} of {}", index, size);

        unsafe { self.unsafe_index(index) }
    }
}

// UnsafeIndex
impl<
    T
> UnsafeIndex<uint, T>
for Vect<T> {
    #[inline]
    unsafe fn unsafe_index<'a>(&'a self, index: &uint) -> &'a T {
        self.as_slice().unsafe_ref(*index)
    }
}
