[![Build Status][status]](https://travis-ci.org/japaric/linalg.rs)

# `linalg.rs`

An **experimental** linear algebra library with [OpenBLAS][blas] acceleration
written in [Rust][rust].

# Here be dragons

Very early stage, API may change, shrink, expand, disappear, etc. without
notice.

# [Documentation][docs]

# Done so far

- Core data structures:
  - `Mat`: Owned matrix
  - `Row`/`Col`: Newtype wrappers to represent row/column vectors
  - `StridedSlice`: A strided slice
  - `View`: Sub-matrix view
- Immutable/mutable views:
  - row
  - column
  - diagonal
  - sub-matrix
  - transpose
- Immutable/mutable iterators
  - over the columns of a (transposed) (sub-)matrix
  - over the elements of a (transposed) (sub-)matrix
  - over the elements of a row/column/diagonal
  - over the rows of a (transposed) (sub-)matrix
- 1D/2D immutable/mutable indexing
- Basic printing of vector/matrices
- Macro sugar to create vector/matrices
- Slicing of vector-like views
- Sum the rows/columns of a (tranposed) (sub-)matrix
- Eager multiplication (Just a few)

# TODO

- `Scaled(scalar, matrix)`
- Pretty printing with proper padding and precision
- Matrix chain multiplication
- Waiting on HKT:
  - Saner traits that don't rely on lifetime lifting
- Waiting on multidispatch:
  - `Equiv`
  - Lots of operators: `+=`, `-=`, `*`, etc

# Ideas to explore

- Convert `Mat` into a fat pointer, then create smart pointers: `Box<Mat>`,
  `Rc<Mat>`, etc.
- Expression templates / lazy evaluation
- `linalg::Error`/`linalg::Result`

# Challenges

- The borrow checker

``` rust
m.mut_row(i) += alpha * m.row(j);  // where i != j
//~^ ERROR cannot borrow `m` as immutable because it is also borrowed as mutable
```

# License

linalg.rs is dual licensed under the Apache 2.0 license and the MIT license.

See LICENSE-APACHE and LICENSE-MIT for more details.

[blas]: https://github.com/xianyi/OpenBLAS
[docs]: http://rust-ci.org/japaric/linalg.rs/doc/linalg/
[rust]: http://www.rust-lang.org/
[status]: https://travis-ci.org/japaric/linalg.rs.svg?branch=master
