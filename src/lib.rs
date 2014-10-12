//! An experimental linear algebra library with BLAS acceleration written in Rust
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
//! ```
//!
//! # Conventions
//!
//! - All operations are [`O(1)`](http://en.wikipedia.org/wiki/Big_O_notation) in time and memory
//!   unless otherwise noted
//! - Matrices are laid in memory using
//!   [column-major order](https://en.wikipedia.org/wiki/Row-major_order)
//! - Element-wise iteration over matrices is done in the fastest way possible, there's no
//!   guarantee of the iteration order

#![deny(warnings)]
#![feature(if_let, macro_rules, phase, tuple_indexing)]

extern crate libc;
extern crate num;
#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[phase(plugin)]
extern crate quickcheck_macros;

use std::kinds::marker;
use std::num::{One, Zero};
use std::rand::distributions::IndependentSample;
use std::rand::{Rand, Rng};

mod blas;
mod col;
mod cols;
mod diag;
mod index;
mod mat;
mod mutcols;
mod mutrows;
mod mutview;
mod notsafe;
mod private;
mod row;
mod rows;
mod show;
mod strided;
#[cfg(test)]
mod test;
mod trans;
mod view;

pub mod traits;

static EXPECT_MSG: &'static str = "capacity overflow";

/// Column vector
///
/// # Restrictions
///
/// - Length is enforced to be greater than one (i.e. not a scalar)
#[deriving(PartialEq)]
pub struct Col<D>(D);

/// Iterator over the columns of an immutable matrix
// TODO (rust-lang/rust#16596) Add a `MatrixCol` bound on `M`
pub struct Cols<'a, M> where M: 'a {
    mat: &'a M,
    state: uint,
    stop: uint,
}

/// View into the diagonal of a matrix
pub struct Diag<D>(D);

/// Owned matrix
///
/// # Restrictions
///
/// - Size is enforced to be at least `(2, 2)` (i.e. not a scalar/vector)
#[deriving(PartialEq)]
pub struct Mat<T> {
    // NB `stride` goes first to optimize the `PartialEq` derived implementation
    stride: uint,
    data: Vec<T>,
}

/// Mutable sub-matrix view
///
/// # Restrictions
///
/// - Size is enforced to be at least `(2, 2)` (i.e. not a scalar/vector)
pub struct MutView<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nocopy: marker::NoCopy,
    _nosend: marker::NoSend,
    data: *mut T,
    size: (uint, uint),
    stride: uint,
}

/// Iterator over the columns of a mutable matrix
// TODO (rust-lang/rust#16596) Add a `MatrixMutCol` bound on `M`
pub struct MutCols<'a, M> where M: 'a {
    mat: &'a mut M,
    state: uint,
    stop: uint,
}

/// Iterator over the rows of a mutable matrix
// TODO (rust-lang/rust#16596) Add a `MatrixMutRow` bound on `M`
pub struct MutRows<'a, M> where M: 'a {
    mat: &'a mut M,
    state: uint,
    stop: uint,
}

/// Row vector
///
/// # Restrictions
///
/// - Length is enforced to be greater than one (i.e. not a scalar)
#[deriving(PartialEq)]
pub struct Row<D>(D);

/// Iterator over the rows of an immutable matrix
// TODO (rust-lang/rust#16596) Add a `MatrixRow` bound on `M`
pub struct Rows<'a, M> where M: 'a {
    mat: &'a M,
    state: uint,
    stop: uint,
}

// NB These Strided* structs have to live here because of visibility
/// Immutable strided slice iterator
pub struct StridedItems<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    state: *const T,
    stride: int,
    stop: *const T,
}

/// Mutable strided slice iterator
pub struct StridedMutItems<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nocopy: marker::NoCopy,
    _nosend: marker::NoSend,
    state: *mut T,
    stride: int,
    stop: *mut T,
}

/// Mutable strided slice
pub struct StridedMutSlice<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nocopy: marker::NoCopy,
    _nosend: marker::NoSend,
    data: *mut T,
    len: uint,
    stride: uint,
}

impl<'a, T> StridedMutSlice<'a, T> {
    fn new(data: *mut T, len: uint, stride: uint) -> StridedMutSlice<'a, T> {
        StridedMutSlice {
            _contravariant: marker::ContravariantLifetime::<'a>,
            _nocopy: marker::NoCopy,
            _nosend: marker::NoSend,
            data: data,
            len: len,
            stride: stride,
        }
    }
}

/// Immutable strided slice
pub struct StridedSlice<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    data: *const T,
    len: uint,
    stride: uint,
}

impl<'a, T> StridedSlice<'a, T> {
    fn new(data: *const T, len: uint, stride: uint) -> StridedSlice<'a, T> {
        StridedSlice {
            _contravariant: marker::ContravariantLifetime::<'a>,
            _nosend: marker::NoSend,
            data: data,
            len: len,
            stride: stride,
        }
    }
}

/// View into the transpose of a matrix
pub struct Trans<M>(M);

/// Immutable sub-matrix view
///
/// # Restrictions
///
/// - Size is enforced to be at least `(2, 2)` (i.e. not a scalar/vector)
pub struct View<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    data: *const T,
    size: (uint, uint),
    stride: uint,
}

// NB These View*Items structs have to live here because of visibility
/// Immutable sub-matrix iterator
pub struct ViewItems<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    data: *const T,
    state: (uint, uint),
    stop: (uint, uint),
    stride: uint,
}

/// Mutable sub-matrix iterator
pub struct ViewMutItems<'a, T> where T: 'a {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nocopy: marker::NoCopy,
    _nosend: marker::NoSend,
    data: *mut T,
    state: (uint, uint),
    stop: (uint, uint),
    stride: uint,
}

impl<T> Col<Vec<T>> {
    /// Creates a column vector from an existing vector
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::new(vec![0i, 1, 2]), mat![0i; 1; 2])
    /// # }
    /// ```
    pub fn new(data: Vec<T>) -> Col<Vec<T>> {
        assert!(data.len() > 1);

        Col(data)
    }

    /// Creates and initializes a column vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
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
    pub fn from_fn(length: uint, op: |uint| -> T) -> Col<Vec<T>> {
        assert!(length > 1);

        Col(Vec::from_fn(length, op))
    }

    /// Creates a column vector and fills it by sampling a random distribution
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn sample<D, R>(length: uint, distribution: &D, rng: &mut R) -> Col<Vec<T>> where
        D: IndependentSample<T>,
        R: Rng,
    {
        assert!(length > 1);

        Col(Vec::from_fn(length, |_| distribution.ind_sample(rng)))
    }
}

impl<T> Col<Vec<T>> where T: Clone {
    /// Constructs a column vector with copies of a value
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
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
    pub fn from_elem(length: uint, value: T) -> Col<Vec<T>> {
        assert!(length > 1);

        Col(Vec::from_elem(length, value))
    }
}

impl<T> Col<Vec<T>> where T: Clone + One {
    /// Constructs a column vector filled with ones
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::ones(3), mat![1i; 1; 1])
    /// # }
    /// ```
    pub fn ones(length: uint) -> Col<Vec<T>> {
        Col::from_elem(length, ::std::num::one())
    }
}

impl<T> Col<Vec<T>> where T: Clone + Zero {
    /// Constructs a column vector filled with zeros
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::zeros(3), mat![0i; 0; 0])
    /// # }
    /// ```
    pub fn zeros(length: uint) -> Col<Vec<T>> {
        Col::from_elem(length, ::std::num::zero())
    }
}

impl<T> Col<Vec<T>> where T: Rand {
    /// Constructs a randomly initialized column vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn rand<R>(length: uint, rng: &mut R) -> Col<Vec<T>> where R: Rng {
        assert!(length > 1);

        Col(Vec::from_fn(length, |_| rng.gen()))
    }
}

impl<T> Mat<T> {
    /// Creates a matrix from an existing vector
    ///
    /// **Note**: Data is considered to be arranged in column-major order
    ///
    /// # Failure
    ///
    /// Fails if `data.len() % nrows != 0`
    pub fn new(data: Vec<T>, nrows: uint) -> Mat<T> {
        assert!(data.len() % nrows == 0);

        Mat {
            data: data,
            stride: nrows,
        }
    }

    /// Creates and initializes a matrix
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Mat;
    /// assert_eq!(Mat::from_fn((2, 2), |i| i), mat![(0, 0), (0, 1); (1, 0), (1, 1)])
    /// # }
    /// ```
    pub fn from_fn((nrows, ncols): (uint, uint), op: |(uint, uint)| -> T) -> Mat<T> {
        assert!(nrows > 1 && ncols > 1);

        let length = nrows.checked_mul(&ncols).expect(EXPECT_MSG);

        let mut data = Vec::with_capacity(length);
        for col in range(0, ncols) {
            for row in range(0, nrows) {
                data.push(op((row, col)))
            }
        }

        Mat {
            data: data,
            stride: nrows,
        }
    }

    /// Creates a matrix and fills it by sampling a random distribution
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    pub fn sample<D, R>(
        (nrows, ncols): (uint, uint),
        distribution: &D,
        rng: &mut R,
    ) -> Mat<T> where
        D: IndependentSample<T>,
        R: Rng,
    {
        assert!(nrows > 1 && ncols > 1);

        let length = nrows.checked_mul(&ncols).expect(EXPECT_MSG);

        Mat {
            data: Vec::from_fn(length, |_| distribution.ind_sample(rng)),
            stride: nrows,
        }
    }
}

impl<T> Mat<T> where T: Clone {
    /// Constructs a matrix with copies of a value
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Mat;
    /// assert_eq!(Mat::from_elem((3, 2), 2), mat![2i, 2; 2, 2; 2, 2])
    /// # }
    /// ```
    pub fn from_elem((nrows, ncols): (uint, uint), value: T) -> Mat<T> {
        assert!(nrows > 1 && ncols > 1);

        let length = nrows.checked_mul(&ncols).expect(EXPECT_MSG);

        Mat {
            data: Vec::from_elem(length, value),
            stride: nrows,
        }
    }
}

impl<T> Mat<T> where T: Clone + One {
    /// Constructs a matrix filled with ones
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Mat;
    /// assert_eq!(Mat::ones((2, 3)), mat![1i, 1, 1; 1, 1, 1])
    /// # }
    /// ```
    pub fn ones(size: (uint, uint)) -> Mat<T> {
        Mat::from_elem(size, ::std::num::one())
    }
}

impl<T> Mat<T> where T: Clone + One + Zero {
    /// Constructs the identity matrix
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Mat;
    /// assert_eq!(Mat::eye((2, 2)), mat![1i, 0; 0, 1])
    /// # }
    /// ```
    pub fn eye((nrows, ncols): (uint, uint)) -> Mat<T> {
        use traits::{MatrixMutDiag, MutIter};

        assert!(nrows > 1 && ncols > 1);

        let mut mat = Mat::from_elem((nrows, ncols), ::std::num::zero());

        // XXX For some reason this doesn't work
        //for x in mat.mut_diag(0).unwrap().mut_iter() {
            //*x = ::std::num::one();
        //}
        {
            let mut d: Diag<::strided::MutSlice<T>> = mat.mut_diag(0).unwrap();
            for x in d.mut_iter() {
                *x = ::std::num::one();
            }
        }

        mat
    }
}

impl<T> Mat<T> where T: Clone + Zero {
    /// Constructs a matrix filled with zeros
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Mat;
    /// assert_eq!(Mat::zeros((2, 3)), mat![0i, 0, 0; 0, 0, 0])
    /// # }
    /// ```
    pub fn zeros(size: (uint, uint)) -> Mat<T> {
        Mat::from_elem(size, ::std::num::zero())
    }
}

impl<T> Mat<T> where T: Rand {
    /// Constructs a randomly initialized matrix
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    pub fn rand<R>((nrows, ncols): (uint, uint), rng: &mut R) -> Mat<T> where R: Rng {
        assert!(nrows > 1 && ncols > 1);

        let length = nrows.checked_mul(&ncols).expect(EXPECT_MSG);

        Mat {
            data: Vec::from_fn(length, |_| rng.gen()),
            stride: nrows,
        }
    }
}

impl<T> Row<Vec<T>> {
    /// Creates a row vector from an existing vector
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::new(vec![0i, 1, 2]), mat![0i, 1, 2])
    /// # }
    /// ```
    pub fn new(data: Vec<T>) -> Row<Vec<T>> {
        assert!(data.len() > 1);

        Row(data)
    }

    /// Creates and initializes a row vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
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
    pub fn from_fn(length: uint, op: |uint| -> T) -> Row<Vec<T>> {
        assert!(length > 1);

        Row(Vec::from_fn(length, op))
    }

    /// Creates a row vector and fills it by sampling a random distribution
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn sample<D, R>(length: uint, distribution: &D, rng: &mut R) -> Row<Vec<T>> where
        D: IndependentSample<T>,
        R: Rng,
    {
        assert!(length > 1);

        Row(Vec::from_fn(length, |_| distribution.ind_sample(rng)))
    }
}

impl<T> Row<Vec<T>> where T: Clone {
    /// Constructs a row vector with copies of a value
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
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
    pub fn from_elem(length: uint, value: T) -> Row<Vec<T>> {
        assert!(length > 1);

        Row(Vec::from_elem(length, value))
    }
}

impl<T> Row<Vec<T>> where T: Clone + One {
    /// Constructs a row vector filled with ones
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::ones(3), mat![1i, 1, 1])
    /// # }
    /// ```
    pub fn ones(length: uint) -> Row<Vec<T>> {
        Row::from_elem(length, ::std::num::one())
    }
}

impl<T> Row<Vec<T>> where T: Clone + Zero {
    /// Constructs a row vector filled with zeros
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)]
    /// # extern crate linalg;
    /// # #[phase(plugin)] extern crate linalg_macros;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::zeros(3), mat![0i, 0, 0])
    /// # }
    /// ```
    pub fn zeros(length: uint) -> Row<Vec<T>> {
        Row::from_elem(length, ::std::num::zero())
    }
}

impl<T> Row<Vec<T>> where T: Rand {
    /// Constructs a randomly initialized row vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn rand<R>(length: uint, rng: &mut R) -> Row<Vec<T>> where R: Rng {
        assert!(length > 1);

        Row(Vec::from_fn(length, |_| rng.gen()))
    }
}
