[![Build Status][status]](https://travis-ci.org/japaric/linalg.rs)

# `linalg.rs`

An **experimental** linear algebra library with BLAS acceleration

# Here be dragons

Very early stage, API may change, shrink, expand, disappear, etc. without
notice.

# [Documentation][docs]

# Done so far

- Core data structures:
  - `Mat`: Owned matrix
  - `Row`/`Col`: Newtype wrappers to represent row/column vectors
  - `strided::Slice`: A strided slice
  - `[Mut]View`: Sub-matrix view
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
- Macro sugar (`mat!`) to create vector/matrices
- Slicing of vectors
- Eager matrix multiplication
- Unsugared `+=`
- Unsugared `-=`
- Generic "cloning" (from views) via `to_owned`
- A "prelude" meant to be glob imported (`use linalg::prelude::*`), that
  contains the most used structs and traits

# License

linalg.rs is dual licensed under the Apache 2.0 license and the MIT license.

See LICENSE-APACHE and LICENSE-MIT for more details.

[docs]: http://rust-ci.org/japaric/linalg.rs/doc/linalg/
[rust]: http://www.rust-lang.org/
[status]: https://travis-ci.org/japaric/linalg.rs.svg?branch=master
