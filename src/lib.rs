//! An *experimental* linear algebra library with BLAS (*) acceleration
//!
//! (*): This library is continuously tested against OpenBLAS and netlib/reference BLAS
//!
//! # Cargo
//!
//! - Cargo.toml
//!
//! ``` ignore
//! [dependencies.linalg]
//! git = "https://github.com/japaric/linalg.rs"
//!
//! [dependencies.linalg_macros]
//! git = "https://github.com/japaric/linalg.rs"
//! ```
//!
//! - Crate file
//!
//! ``` ignore
//! extern crate linalg;
//! #[phase(plugin)]
//! extern crate linalg_macros;
//!
//! use linalg::prelude::*;
//! ```
//!
//! # Design goals
//!
//! - Make memory allocations explicit
//! - Minimize the creation of temporaries
//! - Accelerate (BLAS/SIMD) everything
//!
//! # Conventions
//!
//! - All operations are [`O(1)`](http://en.wikipedia.org/wiki/Big_O_notation) in time and memory
//!   unless otherwise noted
//! - Matrices are laid in memory using
//!   [column-major order](https://en.wikipedia.org/wiki/Row-major_order)
//! - Element-wise iteration over [sub]matrices is done in the fastest way possible, there's no
//!   guarantee of the iteration order

//#![deny(warnings, missing_docs)]
#![feature(macro_rules, slicing_syntax, unboxed_closures)]

extern crate complex;
extern crate libc;
extern crate onezero;

use std::kinds::marker;
use std::num::Int;
use std::rand::distributions::IndependentSample;
use std::rand::{Rand, Rng};

mod add_assign;
mod at;
mod col;
mod cols;
mod error;
mod iter;
mod len;
mod mat;
mod mul;
mod row;
mod rows;
mod show;
mod slice;
mod sub_assign;
mod to_owned;
mod trans;

pub mod blas;
pub mod prelude;
pub mod strided;
pub mod traits;
pub mod view;

/// Column vector
#[deriving(Copy, PartialEq)]
pub struct Col<V>(V);

impl<T> Col<Box<[T]>> {
    /// Creates a column vector from an existing vector
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::new(box [0i, 1, 2]), mat![0i; 1; 2])
    /// # }
    /// ```
    pub fn new(data: Box<[T]>) -> Col<Box<[T]>> {
        Col(data)
    }

    /// Creates a column vector and initializes each element to `f(index)`
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::from_fn(3, |i| i), mat![0; 1; 2])
    /// # }
    /// ```
    pub fn from_fn<F>(length: uint, f: F) -> Col<Box<[T]>> where F: FnMut(uint) -> T {
        Col(Vec::from_fn(length, f).into_boxed_slice())
    }

    /// Creates a column vector and fills it by sampling a random distribution
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn sample<D, R>(length: uint, distribution: &D, rng: &mut R) -> Col<Box<[T]>> where
        D: IndependentSample<T>,
        R: Rng,
    {
        Col(Vec::from_fn(length, |_| distribution.ind_sample(rng)).into_boxed_slice())
    }
}

impl<T> Col<Box<[T]>> where T: Clone {
    /// Constructs a column vector with copies of a value
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::from_elem(3, 2), mat![2i; 2; 2])
    /// # }
    /// ```
    pub fn from_elem(length: uint, value: T) -> Col<Box<[T]>> {
        Col(Vec::from_elem(length, value).into_boxed_slice())
    }
}

impl<T> Col<Box<[T]>> where T: Rand {
    /// Constructs a randomly initialized column vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn rand<R>(length: uint, rng: &mut R) -> Col<Box<[T]>> where R: Rng {
        Col(Vec::from_fn(length, |_| rng.gen()).into_boxed_slice())
    }
}

/// Iterator over the columns of an immutable matrix
#[deriving(Copy)]
pub struct Cols<'a, M> where M: 'a {
    mat: &'a M,
    state: uint,
    stop: uint,
}

/// View into the diagonal of a matrix
#[deriving(Copy)]
pub struct Diag<V>(V);

/// Owned matrix
#[deriving(PartialEq)]
pub struct Mat<T> {
    // NB `size` goes first to optimize the `PartialEq` derived implementation
    size: (uint, uint),
    data: Box<[T]>,
}

impl<T> Mat<T> {
    /// Creates a matrix from a owned buffer and a specified size
    ///
    /// **Note**: `data` is considered to be arranged in column-major order
    ///
    /// # Safety requirements
    ///
    /// - `data.len() == nrows * ncols`
    pub unsafe fn from_parts(data: Box<[T]>, (nrows, ncols): (uint, uint)) -> Mat<T> {
        Mat {
            data: data,
            size: (nrows, ncols),
        }
    }

    /// Creates a matrix and initializes each element to `f(index)`
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Errors
    ///
    /// - `LengthOverflow` if the operation `nrows * ncols` overflows
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Mat;
    /// assert_eq!(Mat::from_fn((2, 2), |i| i).unwrap(), mat![(0, 0), (0, 1); (1, 0), (1, 1)])
    /// # }
    /// ```
    pub fn from_fn<F>((nrows, ncols): (uint, uint), mut f: F) -> Result<Mat<T>> where
        F: FnMut((uint, uint)) -> T,
    {
        let length = match nrows.checked_mul(ncols) {
            Some(length) => length,
            None => return Err(Error::LengthOverflow),
        };

        let mut data = Vec::with_capacity(length);
        for col in range(0, ncols) {
            for row in range(0, nrows) {
                data.push(f((row, col)))
            }
        }

        Ok(Mat {
            data: data.into_boxed_slice(),
            size: (nrows, ncols),
        })
    }

    /// Creates a matrix and fills it by sampling a random distribution
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Errors
    ///
    /// - `LengthOverflow` if the operation `nrows * ncols` overflows
    pub fn sample<D, R>(
        (nrows, ncols): (uint, uint),
        distribution: &D,
        rng: &mut R,
    ) -> Result<Mat<T>> where
        D: IndependentSample<T>,
        R: Rng,
    {
        let length = match nrows.checked_mul(ncols) {
            Some(length) => length,
            None => return Err(Error::LengthOverflow),
        };

        Ok(Mat {
            data: Vec::from_fn(length, |_| distribution.ind_sample(rng)).into_boxed_slice(),
            size: (nrows, ncols),
        })
    }
}

impl<T> Mat<T> where T: Clone {
    /// Constructs a matrix with copies of a value
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Errors
    ///
    /// - `LengthOverflow` if the operation `nrows * ncols` overflows
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Mat;
    /// assert_eq!(Mat::from_elem((3, 2), 2).unwrap(), mat![2i, 2; 2, 2; 2, 2])
    /// # }
    /// ```
    pub fn from_elem((nrows, ncols): (uint, uint), value: T) -> Result<Mat<T>> {
        let length = match nrows.checked_mul(ncols) {
            Some(length) => length,
            None => return Err(Error::LengthOverflow),
        };

        Ok(Mat {
            data: Vec::from_elem(length, value).into_boxed_slice(),
            size: (nrows, ncols),
        })
    }
}

impl<T> Mat<T> where T: Rand {
    /// Constructs a randomly initialized matrix
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Errors
    ///
    /// - `LengthOverflow` if the operation `nrows * ncols` overflows
    pub fn rand<R>((nrows, ncols): (uint, uint), rng: &mut R) -> Result<Mat<T>> where R: Rng {
        let length = match nrows.checked_mul(ncols) {
            Some(length) => length,
            None => return Err(Error::LengthOverflow),
        };

        Ok(Mat {
            data: Vec::from_fn(length, |_| rng.gen()).into_boxed_slice(),
            size: (nrows, ncols),
        })
    }
}

/// Iterator over the columns of a mutable matrix
pub struct MutCols<'a, M> where M: 'a {
    mat: &'a mut M,
    state: uint,
    stop: uint,
}

/// Iterator over the rows of a mutable matrix
pub struct MutRows<'a, M> where M: 'a {
    mat: &'a mut M,
    state: uint,
    stop: uint,
}

/// Mutable sub-matrix view
pub struct MutView<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    ptr: *mut T,
    size: (uint, uint),
    stride: uint,
}

/// Row vector
#[deriving(Copy, PartialEq)]
pub struct Row<V>(V);

impl<T> Row<Box<[T]>> {
    /// Creates a row vector from an existing vector
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::new(box [0i, 1, 2]), mat![0i, 1, 2])
    /// # }
    /// ```
    pub fn new(data: Box<[T]>) -> Row<Box<[T]>> {
        Row(data)
    }

    /// Creates a row vector and initializes each element to `f(index)`
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::from_fn(3, |i| i), mat![0, 1, 2])
    /// # }
    /// ```
    pub fn from_fn<F>(length: uint, f: F) -> Row<Box<[T]>> where F: FnMut(uint) -> T {
        Row(Vec::from_fn(length, f).into_boxed_slice())
    }

    /// Creates a row vector and fills it by sampling a random distribution
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn sample<D, R>(length: uint, distribution: &D, rng: &mut R) -> Row<Box<[T]>> where
        D: IndependentSample<T>,
        R: Rng,
    {
        Row(Vec::from_fn(length, |_| distribution.ind_sample(rng)).into_boxed_slice())
    }
}

impl<T> Row<Box<[T]>> where T: Clone {
    /// Constructs a row vector with copies of a value
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::from_elem(3, 2), mat![2i, 2, 2])
    /// # }
    /// ```
    pub fn from_elem(length: uint, value: T) -> Row<Box<[T]>> {
        Row(Vec::from_elem(length, value).into_boxed_slice())
    }
}

impl<T> Row<Box<[T]>> where T: Rand {
    /// Constructs a randomly initialized row vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn rand<R>(length: uint, rng: &mut R) -> Row<Box<[T]>> where R: Rng {
        Row(Vec::from_fn(length, |_| rng.gen()).into_boxed_slice())
    }
}

/// Iterator over the rows of an immutable matrix
#[deriving(Copy)]
pub struct Rows<'a, M> where M: 'a {
    mat: &'a M,
    state: uint,
    stop: uint,
}

/// View into the transpose of a matrix
#[deriving(Copy)]
pub struct Trans<M>(M);

/// Immutable sub-matrix view
pub struct View<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    ptr: *const T,
    size: (uint, uint),
    stride: uint,
}

impl<'a, T> Copy for View<'a, T> {}

/// Errors
#[deriving(Copy, PartialEq, Show)]
pub enum Error {
    /// Invalid slice range, usually: `start > end`
    InvalidSlice,
    /// Attempted to allocate a matrix bigger that `uint::MAX`
    LengthOverflow,
    /// Attempted to index a non-existent column, i.e. `col >= ncols`
    NoSuchColumn,
    /// Attempted to index a non-existent diagonal
    NoSuchDiagonal,
    /// Attempted to index a non-existent row, i.e. `row >= nrows`
    NoSuchRow,
    /// Attempted to index an element outside of the bounds of the matrix
    OutOfBounds,
}

/// The result of a matrix operation
pub type Result<T> = ::std::result::Result<T, Error>;

/// Constructor trait used to not expose this unsafe constructor in the API
trait Strided<T> {
    /// Creates an strided slice from its parts
    ///
    /// # Safety requirements
    ///
    /// - `ptr` must point to a valid slice with a length of at least
    ///   `if len == 0 { 0 } else { (len - 1) * stride * mem::size_of::<T>() + 1 } `
    /// - Usual aliasing/freezing rules must be enforced by the user
    unsafe fn from_parts(ptr: *const T, len: uint, stride: uint) -> Self;
}
