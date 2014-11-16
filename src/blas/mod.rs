//! BLAS acceleration

use std::num;
use std::raw::Repr;

use {Mat, MutView, View};
use traits::Collection;

pub use self::axpy::Axpy;
pub use self::copy::Copy;
pub use self::dot::Dot;
pub use self::ffi::blasint;
pub use self::gemm::Gemm;
pub use self::gemv::Gemv;

pub mod axpy;
pub mod copy;
pub mod dot;
pub mod ffi;
pub mod gemm;
pub mod gemv;

#[repr(i8)]
/// Transpose matrix before operation?
pub enum Transpose {
    /// Don't transpose
    NoTrans = 110, // 'n'
    /// Transpose
    Trans = 116,  // 't'
}

impl Not<Transpose> for Transpose {
    fn not(&self) -> Transpose {
        match *self {
            NoTrans => Trans,
            Trans => NoTrans,
        }
    }
}

/// Extension trait to convert an integer into a BLAS integer
pub trait ToBlasInt {
    /// Cast the input to `blasint`
    ///
    /// # Panics
    ///
    /// Panics if conversion overflows
    fn to_blasint(self) -> self::ffi::blasint;
}

// FIXME (AI) Use `cast.rs` instead of `num::cast`
impl ToBlasInt for uint {
    fn to_blasint(self) -> self::ffi::blasint {
        num::cast(self).expect("casting to blasint failed")
    }
}

/// Immutable BLAS vector
pub trait Vector<T>: Collection {
    /// Returns a pointer to the start of the memory chunk
    fn as_ptr(&self) -> *const T;
    /// Returns the stride of the vector
    fn stride(&self) -> blasint;

    /// Returns the length of the vector
    fn len(&self) -> blasint {
        Collection::len(self).to_blasint()
    }
}

macro_rules! impl_vector {
    ($($ty:ty),+) => {$(
        impl<'a, T> Vector<T> for $ty {
            fn as_ptr(&self) -> *const T {
                self.repr().data
            }

            fn stride(&self) -> blasint {
                1
            }
        })+
    }
}

impl_vector!(&'a [T], &'a mut [T])

impl<T> Vector<T> for Box<[T]> {
    fn as_ptr(&self) -> *const T {
        self.repr().data
    }

    fn stride(&self) -> blasint {
        1
    }
}

/// Mutable BLAS vector
pub trait MutVector<T>: Vector<T> {
    /// Returns a pointer to the start of the memory chunk
    fn as_mut_ptr(&mut self) -> *mut T;
}

impl<'a, T> MutVector<T> for &'a mut [T] {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.repr().data as *mut T
    }
}

impl<T> MutVector<T> for Box<[T]> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.repr().data as *mut T
    }
}

/// Immutable BLAS matrix
pub trait Matrix<T>: ::traits::Matrix {
    /// Returns pointer to the start of the memory chunk
    fn as_ptr(&self) -> *const T;
    /// Returns stride
    fn stride(&self) -> Option<blasint> { None }
    /// Returns whether `Self` is a transpose view
    fn trans(&self) -> Transpose { NoTrans }
}

impl<T> Matrix<T> for Mat<T> {
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
}

impl<T, M> Matrix<T> for ::Trans<M> where M: Matrix<T> {
    fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    fn stride(&self) -> Option<blasint> {
        self.0.stride()
    }

    fn trans(&self) -> Transpose {
        Trans
    }
}

macro_rules! impl_view {
    ($($ty:ty),+) => {$(
        impl<'a, T> Matrix<T> for $ty {
            fn as_ptr(&self) -> *const T {
                self.ptr as *const T
            }

            fn stride(&self) -> Option<blasint> {
                Some(self.stride.to_blasint())
            }
        }
   )+}
}

impl_view!(View<'a, T>, MutView<'a, T>)
