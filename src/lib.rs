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

#![deny(missing_docs, warnings)]
#![feature(default_type_params, macro_rules)]

extern crate complex;
extern crate libc;
extern crate onezero;

use std::num::Int;
use std::rand::distributions::IndependentSample;
use std::rand::{Rand, Rng};
use std::iter::{range, repeat};
use std::raw::Repr;

use traits::{MatrixCols, MatrixRows};

mod add;
mod add_assign;
mod at;
mod col;
mod cols;
mod eq;
mod error;
mod iter;
mod mat;
mod mul;
mod raw;
mod row;
mod rows;
mod scaled;
mod show;
mod slice;
mod sub;
mod sub_assign;
mod to_owned;
mod trans;
mod view;

pub mod blas;
pub mod prelude;
pub mod strided;
pub mod traits;

/// Immutable view into the column of a matrix
pub struct Col<'a, T: 'a>(strided::Slice<'a, T>);

impl<'a, T> Col<'a, T> {
    /// Returns the length of the column
    pub fn len(&self) -> uint {
        self.0.len()
    }
}

impl<'a, T> Copy for Col<'a, T> {}

/// An owned column vector
pub struct ColVec<T>(Box<[T]>);

impl<T> ColVec<T> {
    /// Creates a column vector from an existing vector
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::ColVec;
    /// assert_eq!(ColVec::new(box [0i, 1, 2]), mat![0i; 1; 2])
    /// # }
    /// ```
    pub fn new(data: Box<[T]>) -> ColVec<T> {
        ColVec(data)
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
    /// # use linalg::ColVec;
    /// assert_eq!(ColVec::from_fn(3, |i| i), mat![0; 1; 2])
    /// # }
    /// ```
    pub fn from_fn<F>(length: uint, f: F) -> ColVec<T> where F: FnMut(uint) -> T {
        ColVec(range(0, length).map(f).collect::<Vec<T>>().into_boxed_slice())
    }

    /// Creates a column vector and fills it by sampling a random distribution
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn sample<D, R>(length: uint, distribution: &D, rng: &mut R) -> ColVec<T> where
        D: IndependentSample<T>,
        R: Rng,
    {
        ColVec(range(0, length).map(|_| distribution.ind_sample(rng)).collect::<Vec<T>>().into_boxed_slice())
    }

    fn as_col(&self) -> Col<T> {
        let std::raw::Slice { data, len } = self.0.repr();

        Col(unsafe { From::parts((
            data,
            len,
            1,
        ))})
    }

    fn as_mut_col(&mut self) -> MutCol<T> {
        let std::raw::Slice { data, len } = self.0.repr();

        MutCol(unsafe { From::parts((
            data,
            len,
            1,
        ))})
    }

    /// Returns the length of the column
    pub fn len(&self) -> uint {
        self.0.len()
    }
}

impl<T> ColVec<T> where T: Clone {
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
    /// # use linalg::ColVec;
    /// assert_eq!(ColVec::from_elem(3, 2), mat![2i; 2; 2])
    /// # }
    /// ```
    pub fn from_elem(length: uint, value: T) -> ColVec<T> {
        ColVec(repeat(value).take(length).collect::<Vec<T>>().into_boxed_slice())
    }
}

impl<T> ColVec<T> where T: Rand {
    /// Constructs a randomly initialized column vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn rand<R>(length: uint, rng: &mut R) -> ColVec<T> where R: Rng {
        ColVec(range(0, length).map(|_| rng.gen()).collect::<Vec<T>>().into_boxed_slice())
    }
}

impl<T> Clone for ColVec<T> where T: Clone {
    fn clone(&self) -> ColVec<T> {
        ColVec(self.0.to_vec().into_boxed_slice())
    }
}

/// Iterator over the columns of an immutable matrix
pub struct Cols<'a, M: 'a>(raw::Cols<'a, M>);

impl<'a, M> Copy for Cols<'a, M> {}

/// Immutable view into the diagonal of a matrix
pub struct Diag<'a, T: 'a>(strided::Slice<'a, T>);

impl<'a, T> Diag<'a, T> {
    /// Returns the length of the diagonal
    pub fn len(&self) -> uint {
        self.0.len()
    }
}

impl<'a, T> Copy for Diag<'a, T> {}

/// Immutable sub-matrix iterator
pub struct Items<'a, T: 'a>(raw::view::Items<'a, T>);

impl<'a, T> Copy for Items<'a, T> {}

/// Owned matrix
pub struct Mat<T> {
    ncols: uint,
    nrows: uint,
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
            ncols: ncols,
            nrows: nrows,
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
            ncols: ncols,
            nrows: nrows,
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
            data: range(0, length).map(|_| distribution.ind_sample(rng)).collect::<Vec<T>>().into_boxed_slice(),
            ncols: ncols,
            nrows: nrows,
        })
    }

    fn as_mut_view(&mut self) -> MutView<T> {
        MutView(unsafe { From::parts((
            self.data.as_ptr(),
            self.nrows,
            self.ncols,
            self.nrows,
        ))})
    }

    fn as_view(&self) -> View<T> {
        View(unsafe { From::parts((
            self.data.as_ptr(),
            self.nrows,
            self.ncols,
            self.nrows,
        ))})
    }

    fn unroll(&self) -> Col<T> {
        Col(unsafe { From::parts((
            self.data.as_ptr(),
            self.nrows * self.ncols,
            1,
        ))})
    }

    fn unroll_mut(&mut self) -> MutCol<T> {
        MutCol(unsafe { From::parts((
            self.data.as_ptr(),
            self.nrows * self.ncols,
            1,
        ))})
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
            data: repeat(value).take(length).collect::<Vec<T>>().into_boxed_slice(),
            ncols: ncols,
            nrows: nrows,
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
            data: range(0, length).map(|_| rng.gen()).collect::<Vec<T>>().into_boxed_slice(),
            ncols: ncols,
            nrows: nrows,
        })
    }
}

impl<T> Clone for Mat<T> where T: Clone {
    fn clone(&self) -> Mat<T> {
        Mat {
            data: self.data.to_vec().into_boxed_slice(),
            ncols: self.ncols,
            nrows: self.nrows,
        }
    }
}

/// Mutable view into the column of a matrix
pub struct MutCol<'a, T: 'a>(strided::MutSlice<'a, T>);

impl<'a, T> MutCol<'a, T> {
    fn as_col(&self) -> Col<T> {
        Col(strided::Slice((self.0).0))
    }

    /// Returns the length of the column
    pub fn len(&self) -> uint {
        self.0.len()
    }
}

/// Iterator over the columns of a mutable matrix
pub struct MutCols<'a, M: 'a>(raw::Cols<'a, M>);

/// Immutable view into the diagonal of a matrix
pub struct MutDiag<'a, T: 'a>(strided::MutSlice<'a, T>);

impl<'a, T> MutDiag<'a, T> {
    fn as_diag(&self) -> Diag<T> {
        Diag(strided::Slice((self.0).0))
    }

    /// Returns the length of the diagonal
    pub fn len(&self) -> uint {
        self.0.len()
    }
}

/// Mutable sub-matrix iterator
pub struct MutItems<'a, T: 'a>(raw::view::Items<'a, T>);

/// Mutable view into the row of a matrix
pub struct MutRow<'a, T: 'a>(strided::MutSlice<'a, T>);

impl<'a, T> MutRow<'a, T> {
    fn as_row(&self) -> Row<T> {
        Row(strided::Slice((self.0).0))
    }

    /// Returns the length of the row
    pub fn len(&self) -> uint {
        self.0.len()
    }
}

/// Iterator over the rows of a mutable matrix
pub struct MutRows<'a, M: 'a>(raw::Rows<'a, M>);

/// Mutable sub-matrix view
pub struct MutView<'a, T: 'a>(raw::View<'a, T>);

impl<'a, T> MutView<'a, T> {
    fn as_view(&self) -> View<T> {
        View(self.0)
    }
}

/// Immutable view into the row of a matrix
pub struct Row<'a, T: 'a>(strided::Slice<'a, T>);

impl<'a, T> Row<'a, T> {
    /// Returns the length of the row
    pub fn len(&self) -> uint {
        self.0.len()
    }
}

impl<'a, T> Copy for Row<'a, T> {}

/// An owned row vector
pub struct RowVec<T>(Box<[T]>);

impl<T> RowVec<T> {
    /// Creates a row vector from an existing vector
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::RowVec;
    /// assert_eq!(RowVec::new(box [0i, 1, 2]), mat![0i, 1, 2])
    /// # }
    /// ```
    pub fn new(data: Box<[T]>) -> RowVec<T> {
        RowVec(data)
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
    /// # use linalg::RowVec;
    /// assert_eq!(RowVec::from_fn(3, |i| i), mat![0, 1, 2])
    /// # }
    /// ```
    pub fn from_fn<F>(length: uint, f: F) -> RowVec<T> where F: FnMut(uint) -> T {
        RowVec(range(0, length).map(f).collect::<Vec<T>>().into_boxed_slice())
    }

    /// Creates a row vector and fills it by sampling a random distribution
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn sample<D, R>(length: uint, distribution: &D, rng: &mut R) -> RowVec<T> where
        D: IndependentSample<T>,
        R: Rng,
    {
        RowVec(range(0, length).map(|_| distribution.ind_sample(rng)).collect::<Vec<T>>().into_boxed_slice())
    }

    fn as_mut_row(&mut self) -> MutRow<T> {
        let std::raw::Slice { data, len } = self.0.repr();

        MutRow(unsafe { From::parts((
            data,
            len,
            1,
        ))})
    }

    fn as_row(&self) -> Row<T> {
        let std::raw::Slice { data, len } = self.0.repr();

        Row(unsafe { From::parts((
            data,
            len,
            1,
        ))})
    }

    /// Returns the length of the row
    pub fn len(&self) -> uint {
        self.0.len()
    }
}

impl<T> RowVec<T> where T: Clone {
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
    /// # use linalg::RowVec;
    /// assert_eq!(RowVec::from_elem(3, 2), mat![2i, 2, 2])
    /// # }
    /// ```
    pub fn from_elem(length: uint, value: T) -> RowVec<T> {
        RowVec(repeat(value).take(length).collect::<Vec<T>>().into_boxed_slice())
    }
}

impl<T> RowVec<T> where T: Rand {
    /// Constructs a randomly initialized row vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn rand<R>(length: uint, rng: &mut R) -> RowVec<T> where R: Rng {
        RowVec(range(0, length).map(|_| rng.gen()).collect::<Vec<T>>().into_boxed_slice())
    }
}

impl<T> Clone for RowVec<T> where T: Clone {
    fn clone(&self) -> RowVec<T> {
        RowVec(self.0.to_vec().into_boxed_slice())
    }
}

/// Iterator over the rows of an immutable matrix
pub struct Rows<'a, M: 'a>(raw::Rows<'a, M>);

impl<'a, M> Copy for Rows<'a, M> {}

/// A lazily scaled matrix
#[derive(Copy)]
pub struct Scaled<T, M>(T, M);

impl<T, M> Scaled<T, M> where M: MatrixCols, T: Clone {
    /// Returns an iterator that yields immutable views into the columns of the matrix
    pub fn cols(&self) -> Scaled<T, Cols<M>> {
        Scaled(self.0.clone(), self.1.cols())
    }
}

impl<T, M> Scaled<T, M> where M: MatrixRows, T: Clone {
    /// Returns an iterator that yields immutable views into each row of the matrix
    pub fn rows(&self) -> Scaled<T, Rows<M>> {
        Scaled(self.0.clone(), self.1.rows())
    }
}

/// View into the transpose of a matrix
#[derive(Copy)]
pub struct Trans<M>(M);

/// Immutable sub-matrix view
pub struct View<'a, T: 'a>(raw::View<'a, T>);

impl<'a, T> Copy for View<'a, T> {}

/// The result of a matrix operation
pub type Result<T> = ::std::result::Result<T, Error>;

/// Errors
#[derive(Copy, PartialEq, Show)]
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

// Private versions of `traits::{At, Slice}` to not expose implementations on stdlib types
// XXX Why do I have to document private traits?
/// Private
trait At<I, T> for Sized? {
    /// private
    fn at(&self, I) -> std::result::Result<&T, error::OutOfBounds>;
}

/// Private
trait Slice<'a, I, S> for Sized? {
    /// Private
    fn slice(&'a self, start: I, end: I) -> Result<S>;
}

// FIXME Use `cast.rs` instead of this trait
/// Private
trait ToBlasint {
    /// Private
    fn to_blasint(self) -> blas::blasint;
}

impl ToBlasint for uint {
    fn to_blasint(self) -> blas::blasint {
        let max: blas::blasint = ::std::num::Int::max_value();

        if self > max as uint {
            panic!("Cast overflow (`uint` -> `blasint`)");
        } else {
            self as blas::blasint
        }
    }
}

// Hack because the intra-crate privacy rules are weird
/// Private
trait From<T> {
    /// Private
    unsafe fn parts(T) -> Self;
}
