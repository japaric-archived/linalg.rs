//! An experimental linear algebra library with OpenBLAS [1] acceleration written in Rust
//!
//! [1] I'm developing this library against OpenBLAS, but since BLAS is a standard, it *should*
//! work with other BLAS implementations
//!
//! # Conventions
//!
//! - All operations are [`O(1)`](http://en.wikipedia.org/wiki/Big_O_notation) in time and memory
//!   unless otherwise noted
//! - Matrices are laid in memory using
//!   [column-major order](https://en.wikipedia.org/wiki/Row-major_order)
//! - Element-wise iteration over matrices is done in the fastest way possible, there's no
//!   guarantee of the iteration order

#![feature(macro_rules, phase)]

extern crate libc;
#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[phase(plugin)]
extern crate quickcheck_macros;

use std::kinds::marker;
use std::num::{One, Zero, mod};
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
pub struct Col<D> {
    data: D,
}

/// Iterator over the columns of an immutable matrix
// TODO (rust-lang/rust#16596) Add a `MatrixCol` bound on `M`
pub struct Cols<'a, M: 'a> {
    mat: &'a M,
    state: uint,
    stop: uint,
}

/// View into the diagonal of a matrix
pub struct Diag<D> {
    data: D,
}

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
pub struct MutView<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nocopy: marker::NoCopy,
    _nosend: marker::NoSend,
    data: *mut T,
    size: (uint, uint),
    stride: uint,
}

/// Iterator over the columns of a mutable matrix
// TODO (rust-lang/rust#16596) Add a `MatrixMutCol` bound on `M`
pub struct MutCols<'a, M: 'a> {
    mat: &'a mut M,
    state: uint,
    stop: uint,
}

/// Iterator over the rows of a mutable matrix
// TODO (rust-lang/rust#16596) Add a `MatrixMutRow` bound on `M`
pub struct MutRows<'a, M: 'a> {
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
pub struct Row<D> {
    data: D,
}

/// Iterator over the rows of an immutable matrix
// TODO (rust-lang/rust#16596) Add a `MatrixRow` bound on `M`
pub struct Rows<'a, M: 'a> {
    mat: &'a M,
    state: uint,
    stop: uint,
}

// NB These Strided* structs have to live here because of visibility
/// Immutable strided slice iterator
pub struct StridedItems<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    state: *const T,
    stride: int,
    stop: *const T,
}

/// Mutable strided slice iterator
pub struct StridedMutItems<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nocopy: marker::NoCopy,
    _nosend: marker::NoSend,
    state: *mut T,
    stride: int,
    stop: *mut T,
}

/// Mutable strided slice
pub struct StridedMutSlice<'a, T: 'a> {
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
pub struct StridedSlice<'a, T: 'a> {
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
pub struct Trans<M> {
    mat: M,
}

/// Immutable sub-matrix view
///
/// # Restrictions
///
/// - Size is enforced to be at least `(2, 2)` (i.e. not a scalar/vector)
pub struct View<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    data: *const T,
    size: (uint, uint),
    stride: uint,
}

// NB These View*Items structs have to live here because of visibility
/// Immutable sub-matrix iterator
pub struct ViewItems<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    data: *const T,
    state: (uint, uint),
    stop: (uint, uint),
    stride: uint,
}

/// Mutable sub-matrix iterator
pub struct ViewMutItems<'a, T: 'a> {
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
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::new(vec![0i, 1, 2]), mat![0i; 1; 2])
    /// # }
    /// ```
    pub fn new(data: Vec<T>) -> Col<Vec<T>> {
        assert!(data.len() > 1);

        Col {
            data: data,
        }
    }

    /// Creates and initializes a column vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::from_fn(3, |i| i), mat![0; 1; 2])
    /// # }
    /// ```
    pub fn from_fn(length: uint, op: |uint| -> T) -> Col<Vec<T>> {
        assert!(length > 1);

        Col {
            data: Vec::from_fn(length, op),
        }
    }

    /// Creates a column vector and fills it by sampling a random distribution
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn sample<D: IndependentSample<T>, R: Rng>(
        length: uint,
        distribution: &D, rng: &mut R,
    ) -> Col<Vec<T>> {
        assert!(length > 1);

        Col {
            data: Vec::from_fn(length, |_| distribution.ind_sample(rng)),
        }
    }
}

impl<T: Clone> Col<Vec<T>> {
    /// Constructs a column vector with copies of a value
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::from_elem(3, 2), mat![2i; 2; 2])
    /// # }
    /// ```
    pub fn from_elem(length: uint, value: T) -> Col<Vec<T>> {
        assert!(length > 1);

        Col {
            data: Vec::from_elem(length, value),
        }
    }
}

impl<T: Clone + One> Col<Vec<T>> {
    /// Constructs a column vector filled with ones
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::ones(3), mat![1i; 1; 1])
    /// # }
    /// ```
    pub fn ones(length: uint) -> Col<Vec<T>> {
        Col::from_elem(length, num::one())
    }
}

impl<T: Clone + Zero> Col<Vec<T>> {
    /// Constructs a column vector filled with zeros
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Col;
    /// assert_eq!(Col::zeros(3), mat![0i; 0; 0])
    /// # }
    /// ```
    pub fn zeros(length: uint) -> Col<Vec<T>> {
        Col::from_elem(length, num::zero())
    }
}

impl<T: Rand> Col<Vec<T>> {
    /// Constructs a randomly initialized column vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn rand<R: Rng>(length: uint, rng: &mut R) -> Col<Vec<T>> {
        assert!(length > 1);

        Col {
            data: Vec::from_fn(length, |_| rng.gen()),
        }
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
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
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
    pub fn sample<D: IndependentSample<T>, R: Rng>(
        (nrows, ncols): (uint, uint),
        distribution: &D, rng: &mut R,
    ) -> Mat<T> {
        assert!(nrows > 1 && ncols > 1);

        let length = nrows.checked_mul(&ncols).expect(EXPECT_MSG);

        Mat {
            data: Vec::from_fn(length, |_| distribution.ind_sample(rng)),
            stride: nrows,
        }
    }
}

impl<T: Clone> Mat<T> {
    /// Constructs a matrix with copies of a value
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
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

impl<T: Clone + One> Mat<T> {
    /// Constructs a matrix filled with ones
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Mat;
    /// assert_eq!(Mat::ones((2, 3)), mat![1i, 1, 1; 1, 1, 1])
    /// # }
    /// ```
    pub fn ones(size: (uint, uint)) -> Mat<T> {
        Mat::from_elem(size, num::one())
    }
}

impl<T: Clone + One + Zero> Mat<T> {
    /// Constructs the identity matrix
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Mat;
    /// assert_eq!(Mat::eye((2, 2)), mat![1i, 0; 0, 1])
    /// # }
    /// ```
    pub fn eye((nrows, ncols): (uint, uint)) -> Mat<T> {
        use traits::{MatrixMutDiag, MutIter};

        assert!(nrows > 1 && ncols > 1);

        let mut mat = Mat::from_elem((nrows, ncols), num::zero());

        // XXX For some reason this doesn't work
        //for x in mat.mut_diag(0).unwrap().mut_iter() {
            //*x = num::one();
        //}
        {
            let mut d: Diag<::strided::MutSlice<T>> = mat.mut_diag(0).unwrap();
            for x in d.mut_iter() {
                *x = num::one();
            }
        }

        mat
    }
}

impl<T: Clone + Zero> Mat<T> {
    /// Constructs a matrix filled with zeros
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Mat;
    /// assert_eq!(Mat::zeros((2, 3)), mat![0i, 0, 0; 0, 0, 0])
    /// # }
    /// ```
    pub fn zeros(size: (uint, uint)) -> Mat<T> {
        Mat::from_elem(size, num::zero())
    }
}

impl<T: Rand> Mat<T> {
    /// Constructs a randomly initialized matrix
    ///
    /// - Memory: `O(nrows * ncols)`
    /// - Time: `O(nrows * ncols)`
    pub fn rand<R: Rng>((nrows, ncols): (uint, uint), rng: &mut R) -> Mat<T> {
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
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::new(vec![0i, 1, 2]), mat![0i, 1, 2])
    /// # }
    /// ```
    pub fn new(data: Vec<T>) -> Row<Vec<T>> {
        assert!(data.len() > 1);

        Row {
            data: data,
        }
    }

    /// Creates and initializes a row vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::from_fn(3, |i| i), mat![0, 1, 2])
    /// # }
    /// ```
    pub fn from_fn(length: uint, op: |uint| -> T) -> Row<Vec<T>> {
        assert!(length > 1);

        Row {
            data: Vec::from_fn(length, op),
        }
    }

    /// Creates a row vector and fills it by sampling a random distribution
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn sample<D: IndependentSample<T>, R: Rng>(
        length: uint,
        distribution: &D, rng: &mut R,
    ) -> Row<Vec<T>> {
        assert!(length > 1);

        Row {
            data: Vec::from_fn(length, |_| distribution.ind_sample(rng)),
        }
    }
}

impl<T: Clone> Row<Vec<T>> {
    /// Constructs a row vector with copies of a value
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::from_elem(3, 2), mat![2i, 2, 2])
    /// # }
    /// ```
    pub fn from_elem(length: uint, value: T) -> Row<Vec<T>> {
        assert!(length > 1);

        Row {
            data: Vec::from_elem(length, value),
        }
    }
}

impl<T: Clone + One> Row<Vec<T>> {
    /// Constructs a row vector filled with ones
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::ones(3), mat![1i, 1, 1])
    /// # }
    /// ```
    pub fn ones(length: uint) -> Row<Vec<T>> {
        Row::from_elem(length, num::one())
    }
}

impl<T: Clone + Zero> Row<Vec<T>> {
    /// Constructs a row vector filled with zeros
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    ///
    /// # Example
    ///
    /// ```
    /// # #![feature(phase)] #[phase(plugin, link)] extern crate linalg;
    /// # fn main() {
    /// # use linalg::Row;
    /// assert_eq!(Row::zeros(3), mat![0i, 0, 0])
    /// # }
    /// ```
    pub fn zeros(length: uint) -> Row<Vec<T>> {
        Row::from_elem(length, num::zero())
    }
}

impl<T: Rand> Row<Vec<T>> {
    /// Constructs a randomly initialized row vector
    ///
    /// - Memory: `O(length)`
    /// - Time: `O(length)`
    pub fn rand<R: Rng>(length: uint, rng: &mut R) -> Row<Vec<T>> {
        assert!(length > 1);

        Row {
            data: Vec::from_fn(length, |_| rng.gen()),
        }
    }
}

#[doc(hidden)]
#[macro_export]
// TODO (rust-lang/rfcs#88) Replace this macro with the `$#(..)` syntax
macro_rules! count_args {
    ($x:expr) => {
        1
    };
    ($x:expr, $($xs:expr),+) => {
        1 + count_args!($($xs),+)
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! push_columns {
    ($data:expr <- $($e:expr);+) => ({
        $($data.push($e);)+
    });
    ($data:expr <- $($x:expr, $($xs:expr),+);+) => ({
        $($data.push($x);)+
        push_columns!($data <- $($($xs),+);+)
    });
}

/// Creates an owned row/column vector or matrix from the arguments
#[macro_export]
macro_rules! mat {
    // Row vector: mat![0, 1, 2]
    ($x:expr, $($xs:expr),+) => ({
        let mut data = Vec::with_capacity(count_args!($x, $($xs),+));
        data.push($x);
        $(data.push($xs);)+

        ::linalg::Row::new(data)
    });
    // Column vector: mat![0; 1; 2]
    ($x:expr; $($xs:expr);+) => ({
        let mut data = Vec::with_capacity(count_args!($x, $($xs),+));
        data.push($x);
        $(data.push($xs);)+

        ::linalg::Col::new(data)
    });
    // Owned matrix: mat![0, 1, 2; 3, 4, 5]
    ($x:expr, $($xs:expr),+; $($y:expr, $($ys:expr),+);+) => ({
        let nrows = count_args!($x, $($y),+);
        let ncols = count_args!($x, $($xs),+);

        // FIXME This should be a compiler error
        assert!($(ncols == count_args!($y, $($ys),+))&&+);

        let mut data = Vec::with_capacity(nrows * ncols);
        push_columns!(data <- $x, $($xs),+; $($y, $($ys),+);+)

        ::linalg::Mat::new(data, nrows)
    });
    // Trailing semicolon
    ($($($e:expr),+);+;) => {
        mat![$($($e),+);+]
    };
}
