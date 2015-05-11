//! A linear algebra library with BLAS and LAPACK acceleration
//!
//! # Using linalg
//!
//! Add these dependencies to your Cargo.toml
//!
//! ``` ignore
//! [dependencies.linalg]
//! features = ["macros"]  # Optional, enables the mat! syntax extension
//! git = "https://github.com/japaric/linalg.rs"
//! ```
//!
//! All the functionality is available via the [prelude] module, glob import it:
//!
//! [prelude]: prelude/index.html
//!
//! ``` ignore
//! // Optional, enables the mat! syntax extension
//! #![plugin(linalg_macros)]
//!
//! extern crate linalg;
//!
//! // Imports extension traits and main `struct`s like `Mat`
//! use linalg::prelude::*;
//! ```
//!
//! # Quick reference
//!
//! For NumPy/Octave users
//!
//! - Matrix initialization
//!
//! ``` ignore
//! // Octave
//! A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
//!
//! // Rust
//! let A = mat![1, 2, 3; 4, 5, 6; 7, 8, 9];
//! ```
//!
//! - Indexing
//!
//! ``` ignore
//! // Python
//! A[1, 2] = 5
//! x = A[3, 4]
//!
//! // Rust
//! A[(1, 2)] = 5;
//! let x = A[(3, 4)];  // or `&A[(3, 4)]` or `&mut A[(3, 4)]`
//! ```
//!
//! - Slicing
//!
//! ``` ignore
//! // Python
//! second_row = A[1, :]
//! third_column = A[:, 2]
//! submat = A[:3, 2:]  // or A[0:3, 2:4]
//!
//! // Rust
//! let second_row = A.row(1);        // or A.slice((1, ..))
//! let third_column = A.col(2);      // or A.slice((.., 2))
//! let submat = A.slice((..3, 2..))  // or A.slice((0..3, 2..4))
//!
//! // NOTE All the operations have a mutable variant, just add a `_mut` suffix.
//! // Example: `let second_row = A.row_mut(1)`;
//! ```
//!
//! - Augmented assignment
//!
//! Increase all the elements of the second row by 1.
//!
//! ``` ignore
//! // Python
//! A[1, :] += 1;
//!
//! // Rust
//! A.row_mut(1).add_assign(1)
//! ```
//!
//! Subtract sub-matrices
//!
//! ``` ignore
//! // Python
//! A[1:3, 2:4] -= B[:2, 1:3]
//!
//! // Rust
//! A.slice_mut((1..3, 2..4)).sub_assign(B.slice((..2, 1..3)));
//! ```
//!
//! - Index assignment
//!
//! Set all the elements of the second column to 0
//!
//! ``` ignore
//! // Python
//! A[:, 1] = 0;
//!
//! // Rust
//! A.col_mut(1).set(0);
//! ```
//!
//! - Copy sub-matrices
//!
//! ``` ignore
//! // Python
//! A[1:3, 1:3] = B[2:4, 3:5]
//!
//! // Rust
//! A.slice_mut((1..3, 1..3)).set(B.slice((2..4, 3..5)));
//! ```
//!
//! - Matrix multiplication
//!
//! ``` ignore
//! // Python
//! D = A.dot(B).dot(C)
//!
//! // Rust
//! let D = A * B * C;
//! ```
//!
//! - Transpose
//!
//! ``` ignore
//! // Python
//! A = B.T
//!
//! // Rust
//! let A = B.t();
//! ```
//!
//! - Matrix inverse
//!
//! ``` ignore
//! // Python
//! B = numpy.linalg.inv(A)
//!
//! // Rust
//! let B = A.inv();
//! ```
//!
//! # Overview of the API
//!
//! - There are two types of structures provided by this crate: "owned structures", and "views".
//! The former own the data, and are in charge of freeing memory when `drop`ed, the latter are
//! borrows of the former, and provide limited access to the data.
//! - The owned structures: `Mat`, `ColVec` and `RowVec` provide constructors via static methods.
//! - Most of the functionality (other than overloaded operators) is provided as methods via
//! extension traits. All the public extension traits are grouped in the [traits] module.
//!
//! [traits]: traits/index.html
//!
//! - There are several conversions between views and from views to owned structures available via
//! the `From`/`Into` traits.
//!
//! # Notes about operators
//!
//! - Keep in mind that all operators (unary/binary) take their operands by value.
//!
//! - Both forms of multiplication: matrix multiplication and scaling are lazy. If you want eager
//! evaluation, you can use the `eval()` method, but this allocates memory, and should be avoided
//! whenever possible.
//!
//! ``` ignore
//! // No op
//! let C = A * B;
//!
//! // Performs the matrix multiplication
//! let C = (A * B).eval();
//! ```
//!
//! - When multiplying matrices, you never want to give up ownership. Always multiply views.
//!
//! ``` ignore
//! // A and B are owned
//! let C = A * B;  //~ error
//!
//! let C = &A * &B;  // OK
//! ```
//!
//! - Addition and subtraction are eager, and one of the operands must provide a buffer to store
//! the result, in most cases this means that one of the operators needs to be *moved into* the
//! operation.
//!
//! ``` ignore
//! // Both A and B own their data
//!
//! let C = A + &B;  // OK, result will be stored in A's buffer
//!
//! let C = A + B;
//! //~^ error: the result could be stored in A's buffer, but B would be unnecessarily dropped
//!
//! let C = &A + &B;  //~ error: nowhere to store the result
//! ```
//!
//! - When you only have views and need to perform an addition/subtraction, you'll have to allocate
//! memory. For performance, avoid allocating in loops.
//!
//! ``` ignore
//! // theta, X and y are views
//!
//! // Bad: allocation in a loop
//! loop {
//!     let y = ColVec::from(y);  // Deep copy
//!     let e = y - theta * X;
//!
//!     ..
//!
//!     if condition(&e) { break }
//! }
//!
//! // Instead allocate a buffer outside the loop, and re-use it in the loop
//! let mut z = ColVec::zeros(y.nrows());
//!
//! loop {
//!     // z = y
//!     z.set(y);
//!
//!     // z = y - theta * X
//!     z.sub_assign(theta * X);
//!
//!     ..
//!
//!     if condition(&z) { break }
//! }
//! ```
//!
//! - Transposing a matrix is "free", no allocations, copies or operations are performed . Do note
//! that the `t()` method takes the caller by value.
//!
//! - The `inv()` method computes the inverse of an owned (square) matrix and takes ownership of
//! the caller. The caller's buffer will be re-used to store the inverse.

#![deny(missing_docs)]
#![deny(warnings)]
#![feature(advanced_slice_patterns)]
#![feature(collections)]
#![feature(core)]
#![feature(filling_drop)]
#![feature(slice_patterns)]
#![feature(unique)]
#![feature(unsafe_no_drop_flag)]

extern crate assign;
extern crate blas;
extern crate cast;
extern crate complex;
extern crate core;
extern crate extract;
extern crate lapack;
extern crate onezero;

mod chain;
mod cols;
mod debug;
mod linear;
mod mat;
mod ops;
mod product;
mod rows;
mod scaled;
mod stripes;
mod submat_mut;
mod tor;

pub mod prelude;
pub mod strided;
pub mod submat;
pub mod traits;
pub mod transposed;

use core::nonzero::NonZero;
use std::marker::PhantomData;
use std::ops::{Range, RangeFull};
use std::ptr::Unique;
use std::{mem, slice};

use blas::Transpose;
use cast::From as _0;
use extract::Extract;

use traits::Matrix;

/// A reserved chunk of memory
pub struct Buffer<T>(Vec<T>);

impl<T> Buffer<T> {
    /// Creates a buffer with size `n`
    pub fn new(n: usize) -> Buffer<T> where T: Copy {
        unsafe {
            let mut v = Vec::with_capacity(n);
            v.set_len(n);

            Buffer(v)
        }
    }

    /// Exposes this buffer as a pool of matrices
    pub fn as_pool(&mut self) -> Pool<T> {
        Pool(Some(&mut self.0[..]))
    }
}

/// Lazy matrix chain multiplication
pub struct Chain<'a, T> {
    first: (Transpose, SubMat<'a, T>),
    second: (Transpose, SubMat<'a, T>),
    tail: Vec<(Transpose, SubMat<'a, T>)>,
}

impl<'a, T> Chain<'a, T> {
    fn len(&self) -> usize {
        self.tail.len() + 2
    }
}

/// Immutable view into the column of a matrix
pub struct Col<'a, T>(Slice<'a, T>);

/// Mutable "view" into the column of a matrix
pub struct ColMut<'a, T>(Col<'a, T>);

/// Owned column vector
#[derive(Clone)]
pub struct ColVec<T>(Tor<T>);

/// Iterator over the columns of an immutable matrix
pub struct Cols<'a, T>(SubMat<'a, T>);

/// Iterator over the columns of a mutable matrix
pub struct ColsMut<'a, T>(Cols<'a, T>);

/// An immutable view into the diagonal of a matrix
pub struct Diag<'a, T>(Slice<'a, T>);

/// A mutable "view" into the diagonal of a matrix
pub struct DiagMut<'a, T>(Diag<'a, T>);

/// An immutable iterator over a matrix in horizontal stripes
pub struct HStripes<'a, T> {
    mat: SubMat<'a, T>,
    size: i32,
}

/// A "mutable" iterator over a matrix in horizontal stripes
pub struct HStripesMut<'a, T>(HStripes<'a, T>);

/// An owned matrix
// NB `nrows` and `ncols` are guaranteed to be non-negative
#[unsafe_no_drop_flag]
pub struct Mat<T> {
    data: Unique<T>,
    ncols: i32,
    nrows: i32,
}

impl<T> Mat<T> {
    unsafe fn uninitialized((nrows, ncols): (i32, i32)) -> Mat<T> {
        debug_assert!(ncols >= 0);
        debug_assert!(nrows >= 0);

        let nrows_ = usize::from_(nrows).extract();
        let ncols_ = usize::from_(ncols).extract();

        debug_assert!(nrows_.checked_mul(ncols_).is_some());

        let n = nrows_ * ncols_;

        let mut v = Vec::with_capacity(n);
        let data = v.as_mut_ptr();
        mem::forget(v);

        Mat {
            data: Unique::new(data),
            nrows: nrows,
            ncols: ncols,
        }
    }

    fn as_slice(&self) -> &[T] {
        unsafe {
            let len = usize::from_(self.nrows).extract() * usize::from_(self.ncols).extract();

            slice::from_raw_parts(*self.data, len)
        }
    }

}

/// A pool of uninitialized matrices
pub struct Pool<'a, T>(Option<&'a mut [T]>) where T: 'a;

impl<'a, T> Pool<'a, T> {
    /// Returns an uninitialized column vector of size `n`
    pub fn col(&mut self, n: u32) -> ColMut<'a, T> {
        unsafe {
            let slice = self.0.take().extract();

            let at = usize::from_(n);
            let (left, right) = slice.split_at_mut(at);

            self.0 = Some(right);

            ColMut::from(left)
        }
    }

    /// Returns an uninitialized matrix of size `(nrows, ncols)`
    pub fn mat(&mut self, (nrows, ncols): (u32, u32)) -> SubMatMut<'a, T> {
        unsafe {
            let slice = self.0.take().extract();

            let at = usize::from_(nrows) * usize::from_(ncols);
            let (left, right) = slice.split_at_mut(at);

            self.0 = Some(right);

            SubMatMut::reshape(left, (nrows, ncols))
        }
    }

    /// Returns an uninitialized row vector of size `n`
    pub fn row(&mut self, n: u32) -> RowMut<'a, T> {
        unsafe {
            let slice = self.0.take().extract();

            let at = usize::from_(n);
            let (left, right) = slice.split_at_mut(at);

            self.0 = Some(right);

            RowMut::from(left)
        }
    }
}

/// Lazy matrix product
// NB Combinations:
//
// - Col-like: `Product<Chain, Col>`, `Product<Transposed<SubMat>, Col>`, `Product<SubMat, Col>`
// - Row-like: `Product<Row, Chain>`, `Product<Row, Transposed<SubMat>>`, `Product<Row, SubMat>`
//
// -> 6 types
pub struct Product<L, R>(L, R);

/// Immutable view into the row of a matrix
pub struct Row<'a, T>(Slice<'a, T>);

/// Mutable "view" into the row of a matrix
pub struct RowMut<'a, T>(Row<'a, T>);

/// An owned row vector
#[derive(Clone)]
pub struct RowVec<T>(Tor<T>);

/// Iterator over the rows of an immutable matrix
pub struct Rows<'a, T>(SubMat<'a, T>);

/// Iterator over the rows of a mutable matrix
pub struct RowsMut<'a, T>(Rows<'a, T>);

/// A lazily scaled matrix
// NB `M` can only be `Col`, `Product`, `Row`, `Transposed<SubMat>` or `SubMat`
#[derive(Clone, Copy, Debug)]
pub struct Scaled<M>(M::Elem, M) where M: Matrix;

/// A lazily transposed matrix
// NB `M` can only be `Mat`, `SubMat`, or `SubMatMut`
#[derive(Clone, Copy)]
pub struct Transposed<M>(M);

/// An immutable iterator over a matrix in vertical stripes
pub struct VStripes<'a, T> {
    mat: SubMat<'a, T>,
    size: i32,
}

/// A "mutable" iterator over a matrix in vertical stripes
pub struct VStripesMut<'a, T>(VStripes<'a, T>);

/// Immutable sub-matrix view
// NB `ncols`, `nrows` and `stride` are guaranteed to be non-negative
pub struct SubMat<'a, T> {
    _marker: PhantomData<fn() -> &'a T>,
    data: NonZero<*mut T>,
    ncols: i32,
    nrows: i32,
    stride: i32,
}

impl<'a, T> SubMat<'a, T> {
    unsafe fn new(data: *mut T, (nrows, ncols): (i32, i32), stride: i32) -> SubMat<'a, T> {
        debug_assert!(ncols >= 0);
        debug_assert!(nrows >= 0);
        debug_assert!(stride >= 0);

        SubMat {
            _marker: PhantomData,
            data: NonZero::new(data),
            ncols: ncols,
            nrows: nrows,
            stride: stride,
        }
    }

    /// Reshapes a slice as an immutable (sub)matrix
    ///
    /// # Panics
    ///
    /// If:
    ///
    /// - `nrows * ncols != slice.len() ||`
    /// - `nrows > 2^31 ||`
    /// - `ncols > 2^31 ||`
    pub fn reshape(slice: &[T], (nrows, ncols): (u32, u32)) -> SubMat<T> {
        unsafe {
            assert_eq!(slice.len(), usize::from_(nrows) * usize::from_(ncols));

            let nrows = i32::from_(nrows).unwrap();
            let ncols = i32::from_(ncols).unwrap();
            let data = slice.as_ptr() as *mut T;
            let stride = nrows;

            SubMat::new(data, (nrows, ncols), stride)
        }
    }

    fn as_slice(&self) -> Option<&[T]> {
        unsafe {
            if self.nrows == self.stride {
                let len = usize::from_(self.nrows).extract() * usize::from_(self.ncols).extract();

                Some(slice::from_raw_parts(*self.data, len))
            } else {
                None
            }
        }
    }

    unsafe fn raw_index(&self, (row, col): (u32, u32)) -> *mut T {
        assert!(row < self.nrows() && col < self.ncols());

        self.unsafe_index((i32::from_(row).extract(), i32::from_(col).extract()))
    }

    unsafe fn unsafe_col(&self, i: i32) -> Col<'a, T> {
        debug_assert!(i >= 0);
        debug_assert!(i < self.ncols);

        let data = self.data.offset(isize::from_(i) * isize::from_(self.stride));
        let len = self.nrows;
        let stride = 1;

        Col(Slice::new(data, len, stride))
    }

    unsafe fn unsafe_index(&self, (row, col): (i32, i32)) -> *mut T {
        debug_assert!(row < self.nrows);
        debug_assert!(col < self.ncols);

        self.data.offset(isize::from_(col) * isize::from_(self.stride) + isize::from_(row))
    }

    unsafe fn unsafe_row(&self, i: i32) -> Row<'a, T> {
        debug_assert!(i >= 0);
        debug_assert!(i < self.nrows);

        let data = self.data.offset(isize::from_(i));
        let len = self.ncols;
        let stride = self.stride;

        Row(Slice::new(data, len, stride))
    }

    unsafe fn unsafe_slice(
        &self,
        Range { start: (srow, scol), end: (erow, ecol) }: Range<(i32, i32)>,
    ) -> SubMat<'a, T> {
        debug_assert!(srow >= 0);
        debug_assert!(srow <= erow);
        debug_assert!(erow <= self.nrows);
        debug_assert!(scol >= 0);
        debug_assert!(scol <= ecol && ecol <= self.ncols);
        debug_assert!(ecol <= self.ncols);

        let stride = self.stride;
        let data = self.data.offset(isize::from_(scol) * isize::from_(stride) + isize::from_(srow));

        SubMat::new(data, (erow - srow, ecol - scol), stride)
    }

    unsafe fn unsafe_hsplit_at(&self, i: i32) -> (SubMat<'a, T>, SubMat<'a, T>) {
        debug_assert!(i >= 0);
        debug_assert!(i <= self.nrows);

        let ncols = self.ncols;
        let nrows = self.nrows;

        (self.unsafe_slice((0, 0)..(i, ncols)), self.unsafe_slice((i, 0)..(nrows, ncols)))
    }

    unsafe fn unsafe_vsplit_at(&self, i: i32) -> (SubMat<'a, T>, SubMat<'a, T>) {
        debug_assert!(i >= 0);
        debug_assert!(i <= self.ncols);

        let ncols = self.ncols;
        let nrows = self.nrows;

        (self.unsafe_slice((0, 0)..(nrows, i)), self.unsafe_slice((0, i)..(nrows, ncols)))
    }
}

/// Mutable sub-matrix "view"
pub struct SubMatMut<'a, T>(SubMat<'a, T>);

impl<'a, T> SubMatMut<'a, T> {
    /// Reshapes a slice as a mutable (sub)matrix
    ///
    /// # Panics
    ///
    /// If:
    ///
    /// - `nrows * ncols != slice.len() ||`
    /// - `nrows > 2^31 ||`
    /// - `ncols > 2^31 ||`
    pub fn reshape(slice: &mut [T], (nrows, ncols): (u32, u32)) -> SubMatMut<T> {
        unsafe {
            assert_eq!(slice.len(), usize::from_(nrows) * usize::from_(ncols));

            let nrows = i32::from_(nrows).unwrap();
            let ncols = i32::from_(ncols).unwrap();
            let data = slice.as_mut_ptr();
            let stride = nrows;

            SubMatMut(SubMat::new(data, (nrows, ncols), stride))
        }
    }

    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.0.as_slice().map(|s| unsafe {
            slice::from_raw_parts_mut(s.as_ptr() as *mut _, s.len())
        })
    }
}

/// Strided slice
// `len` is guaranteed to be non-negative, and `stride` is guaranteed to be positive
struct Slice<'a, T> {
    _marker: PhantomData<fn() -> &'a T>,
    data: NonZero<*mut T>,
    len: i32,
    stride: NonZero<i32>,
}

/// Owned slice with `i32` length
// NB `len` guaranteed to be non-negative
#[unsafe_no_drop_flag]
struct Tor<T> {
    data: Unique<T>,
    len: i32,
}

trait Forward: Sized {
    fn slice(self, _: RangeFull) -> Self {
        self
    }
}

impl<'a, T> Forward for Chain<'a, T> {}
impl<L, R> Forward for Product<L, R> {}
impl<M> Forward for Scaled<M> where M: Matrix {}

macro_rules! copy {
    ($($ty:ident),+) => {
        $(
            impl<'a, T> Clone for $ty<'a, T> {
                fn clone(&self) -> $ty<'a, T> {
                    *self
                }
            }

            impl<'a, T> Copy for $ty<'a, T> {}
         )+
    };
}

copy!(Col, Diag, Row, Slice, SubMat);

macro_rules! send {
    ($($ty:ident),+) => {
        $(
            unsafe impl<'a, T> Send for $ty<'a, T> where T: Sync {}
         )+
    };
}

send!(Col, Row, SubMat);

macro_rules! send_mut {
    ($($ty:ident),+) => {
        $(
            unsafe impl<'a, T> Send for $ty<'a, T> where T: Send {}
         )+
    };
}

send_mut!(ColMut, RowMut, SubMatMut);
