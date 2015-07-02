//! Linear algebra
//!
//! # Quick reference
//!
//! For NumPy/Octave users
//!
//! - Matrix initialization
//!
//! NOTE Not yet ported from the old linalg version
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
//! A[1, 2] = 5;
//! let x = A[3, 4];  // or `&A[3, 4]` or `&mut A[3, 4]`
//! ```
//!
//! - Slicing
//!
//! ``` ignore
//! // Python
//! second_row = A[1, :]
//! third_column = A[:, 2]
//! submat = A[:3, 2:]  // or `A[0:3, 2:4]`
//!
//! // Rust
//! let second_row = &A[1]; // or `&A[1, ..]`
//! let third_column = &A[.., 2];
//! let submat = &A[..3, 2..];  // or `A[0..3, 2..4]`
//!
//! // NOTE All the operations have a mutable variant, just change `&` with `&mut`
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
//! A[1, ..] += 1;
//! ```
//!
//! Subtract sub-matrices
//!
//! ``` ignore
//! // Python
//! A[1:3, 2:4] -= B[:2, 1:3]
//!
//! // Rust
//! A[1..3, 2..4] -= B[..2, 1..3];
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
//! A[.., 1] = 0;
//! ```
//!
//! - Copy sub-matrices
//!
//! ``` ignore
//! // Python
//! A[1:3, 1:3] = B[2:4, 3:5]
//!
//! // Rust
//! A[1..3, 1..3] = B[2..4, 3..5];
//! ```
//!
//! - Matrix multiplication
//!
//! ``` ignore
//! // Python
//! D = A.dot(B).dot(C)
//!
//! // Rust
//! D[..] = A * B * C;
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
//! NOTE Not yet ported from the old linalg version
//!
//! ``` ignore
//! // Python
//! B = numpy.linalg.inv(A)
//!
//! // Rust
//! let B = A.inv();
//! ```
//!
//! # Notes about arithmetic operations
//!
//! - Operations are lazy when all the arguments are passed by immutable reference (`&Mat`,
//! `&Row`). `A * B` returns a proxy when e.g. `A` and `B` have type `&Mat`.
//! - To force evaluation use the indexed assignment operation: `A[..] = B * C`. The result will be
//! stored in the LHS (`A`), and no allocation will be performed during the execution ("in most
//! cases" -- chain products of 3 or more matrices `A * B * C` are one exception).
//! - Operations are eager when exactly one of the arguments can be used as output buffer, e.g.
//! `let Z = alpha * A + B`, where `A: &Mat` and `B: &mut Mat` -- `B` will be consumed by the
//! operation and its buffer will be reused to store the result. In other words you can think of
//! the previous operation as sugar for `let Z = { B += alpha * A; B }`.
//! - Operations that would consume more buffers than necessary (because of move semantics) will be
//! rejected at compile time. e.g. `Box<Mat> + Box<Mat>`
//! - The transpose operator `t()` is zero cost, no deep copy (clone) or allocation is performed
//! when its called. Do note that it takes the caller by value, so both `Box<Mat>` and `&mut Mat`
//! will be moved.

//#![deny(warnings)]
#![deny(missing_docs)]

#![feature(advanced_slice_patterns)]
#![feature(augmented_assignments)]
#![feature(box_raw)]
#![feature(core)]
#![feature(indexed_assignment)]
#![feature(into_cow)]
#![feature(raw)]
#![feature(slice_patterns)]
#![feature(unsized_types)]
#![feature(zero_one)]

// nn example
#![feature(scoped)]

extern crate blas;
extern crate cast;
extern crate extract;

#[macro_use]
extern crate log;

mod col;
mod iter;
mod mat;
mod row;
mod vector;

mod nn;

fn main() {
    nn::main();
}

pub mod ops;
pub mod order;
pub mod prelude;
pub mod strided;
pub mod traits;
pub mod u31;

use traits::Scalar;

/// A reserved chunk of memory
pub struct Buffer<T>(Vec<T>);

impl<T> Buffer<T> {
    /// Creates a buffer with size `n`
    pub fn new(n: usize) -> Buffer<T> where T: Scalar {
        unsafe {
            let mut v = Vec::with_capacity(n);
            v.set_len(n);
            Buffer(v)
        }
    }

    /// Exposes this buffer as a pool of matrices
    pub fn as_pool(&mut self) -> Pool<T> {
        Pool(&mut self.0)
    }
}

/// A column vector
#[derive(Debug)]
pub struct Col<T>(Vector<T>);

/// Column-by-column iterator
pub struct Cols<'a, T: 'a, O: 'a> {
    m: &'a ::strided::Mat<T, O>
}

/// Mutable column-by-column iterator
pub struct ColsMut<'a, T: 'a, O: 'a> {
    m: &'a mut ::strided::Mat<T, O>
}

/// Iterator over a matrix in horizontal (non-overlapping) stripes
pub struct HStripes<'a, T: 'a> {
    m: &'a ::Mat<T, ::order::Row>,
    size: u32,
}

/// Iterator over a matrix in horizontal (non-overlapping) mutable stripes
pub struct HStripesMut<'a, T: 'a> {
    m: &'a mut ::Mat<T, ::order::Row>,
    size: u32,
}

/// A matrix
pub unsized type Mat<T, O>;

/// A pool of uninitialized matrices
pub struct Pool<'a, T>(&'a mut [T]) where T: 'a;

impl<'a, T> Pool<'a, T> {
    /// Returns an uninitialized column vector of size `n`
    pub fn col(&mut self, n: u32) -> &'a mut ::Col<T> {
        let len = usize::from(n);
        let tmp = mem::replace(&mut self.0, &mut []);
        let (slice, left) = tmp.split_at_mut(len);
        self.0 = left;

        Col::new_mut(slice)
    }

    /// Returns an uninitialized matrix of size `(nrows, ncols)`
    pub fn mat<O>(&mut self, (nrows, ncols): (u32, u32)) -> &'a mut ::Mat<T, O> where
        O: ::order::Order,
    {
        let len = usize::from(nrows) * usize::from(ncols);
        let tmp = mem::replace(&mut self.0, &mut []);
        let (slice, left) = tmp.split_at_mut(len);
        self.0 = left;

        ::Mat::reshape_mut(slice, (nrows, ncols))
    }

    /// Returns an uninitialized row vector of size `n`
    pub fn row(&mut self, n: u32) -> &'a mut ::Row<T> {
        let len = usize::from(n);
        let tmp = mem::replace(&mut self.0, &mut []);
        let (slice, left) = tmp.split_at_mut(len);
        self.0 = left;

        Row::new_mut(slice)
    }
}

/// A row vector
#[derive(Debug)]
pub struct Row<T>(Vector<T>);

/// Row-by-row iterator
pub struct Rows<'a, T: 'a, O: 'a> {
    m: &'a ::strided::Mat<T, O>,
}

/// Mutable row-by-row iterator
pub struct RowsMut<'a, T: 'a, O: 'a> {
    m: &'a mut ::strided::Mat<T, O>
}

unsized type Vector<T>;

/// Iterator over a matrix in vertical (non-overlapping) stripes
pub struct VStripes<'a, T: 'a> {
    m: &'a ::Mat<T, ::order::Col>,
    size: u32,
}

/// Iterator over a matrix in vertical (non-overlapping) mutable stripes
pub struct VStripesMut<'a, T: 'a> {
    m: &'a mut ::Mat<T, ::order::Col>,
    size: u32,
}

/// Matrix "order", only known at runtime
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Order {
    /// Column major order
    Col,
    /// Row major order
    Row,
}

// NB All the following items are here to avoid leaking implementation details into the public API
use std::marker::PhantomData;
use std::num::Zero;
use std::raw::FatPtr;
use std::{fat_ptr, mem};

use cast::From;

use u31::U31;

impl<T, O> ::Mat<T, O> {
    fn empty<'a>() -> &'a mut ::Mat<T, O> {
        let _0 = U31::zero();

        unsafe {
            &mut *fat_ptr::new(FatPtr {
                data: 1 as *mut T,
                info: ::mat::Info {
                    _marker: PhantomData,
                    ncols: _0,
                    nrows: _0,
                }
            })
        }
    }
}

impl<T, O> ::strided::Mat<T, O> {
    fn empty<'a>() -> &'a mut ::strided::Mat<T, O> {
        let _0 = U31::zero();

        unsafe {
            &mut *fat_ptr::new(FatPtr {
                data: 1 as *mut T,
                info: ::strided::mat::Info {
                    _marker: PhantomData,
                    ncols: _0,
                    nrows: _0,
                    stride: _0,
                }
            })
        }
    }

    fn is_empty(&self) -> bool {
        let _0 = U31::zero();
        let ::strided::mat::Info { nrows, ncols, .. } = self.repr().info;

        nrows == _0 || ncols == _0
    }
}
