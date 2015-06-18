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

#![deny(missing_docs)]

#![feature(advanced_slice_patterns)]
#![feature(augmented_assignments)]
#![feature(core)]
#![feature(filling_drop)]
#![feature(indexed_assignment)]
#![feature(into_cow)]
#![feature(slice_patterns)]
#![feature(unsized_types)]
#![feature(zero_one)]

// nn example
#![feature(scoped)]

extern crate blas;
extern crate cast;
extern crate core;
extern crate extract;

#[macro_use]
extern crate log;

mod col;
mod iter;
mod mat;
mod row;

mod nn;

fn main() {
    nn::main();
}

pub mod ops;
pub mod order;
pub mod prelude;
pub mod raw;
pub mod strided;
pub mod traits;
pub mod u31;

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
        Pool(&mut self.0)
    }
}

/// A column vector
pub unsized type Col<T> = ::raw::Slice<T>;

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
pub unsized type Mat<T, O> = ::raw::Mat<T, O>;

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
pub unsized type Row<T> = ::raw::Slice<T>;

/// Row-by-row iterator
pub struct Rows<'a, T: 'a, O: 'a> {
    m: &'a ::strided::Mat<T, O>,
}

/// Mutable row-by-row iterator
pub struct RowsMut<'a, T: 'a, O: 'a> {
    m: &'a mut ::strided::Mat<T, O>
}

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

// FIXME This should be private
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Order {
    Col,
    Row,
}

// NB All the following items are here to avoid leaking implementation details into the public API
use std::marker::PhantomData;
use std::num::Zero;
use std::ops::Range;
use std::slice;
use std::mem;

use cast::From;
use core::nonzero::NonZero;
use extract::Extract;

use u31::U31;

impl<T> ::raw::Slice<T> {
    unsafe fn from(slice: &[T]) -> ::raw::Slice<T> {
        let len = U31::from(slice.len()).unwrap();
        let data = NonZero::new(slice.as_ptr() as *mut T);

        ::raw::Slice { data: data, len: len }
    }

    unsafe fn as_slice_raw(&self) -> *mut [T] {
        slice::from_raw_parts_mut(*self.data, self.len.usize())
    }

    // NOTE Core
    fn slice(&self, r: Range<u32>) -> ::raw::Slice<T> {
        unsafe {
            assert!(r.start <= r.end);
            assert!(r.end <= self.len.u32());

            ::raw::Slice {
                data: NonZero::new(self.data.offset(r.start as isize)),
                len: U31::from(r.end - r.start).extract(),
            }
        }
    }
}

impl<T, O> ::Mat<T, O> {
    fn empty<'a>() -> &'a mut ::Mat<T, O> {
        unsafe {
            mem::transmute(::raw::Mat {
                data: NonZero::new(1 as *mut T),
                marker: PhantomData::<O>,
                ncols: U31::zero(),
                nrows: U31::zero(),
            })
        }
    }
}

impl<T, O> ::strided::Mat<T, O> {
    fn empty<'a>() -> &'a mut ::strided::Mat<T, O> {
        unsafe {
            mem::transmute(::strided::raw::Mat {
                data: NonZero::new(1 as *mut T),
                marker: PhantomData::<O>,
                ncols: U31::zero(),
                nrows: U31::zero(),
                stride: U31::zero(),
            })
        }
    }

    fn is_empty(&self) -> bool {
        let ::strided::raw::Mat { nrows, ncols, .. } = self.repr();
        nrows == U31::zero() || ncols == U31::zero()
    }
}
