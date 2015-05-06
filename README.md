# `linalg.rs`

Linear algebra library with BLAS and LAPACK acceleration.

## Features

- Dynamically sized dense matrices and row/column vectors
- Expression simplifier: `alpha * A * B + beta * C` reduces to a single [GEMM] call, and doesn't
  allocate memory for the return value.
- [MCOP] solver: The evaluation order of `A * B * .. * Z` is chosen at runtime to minimize the
  number of operations and of temporary memory allocations.
- Matrix inversion and zero-cost matrix transposing
- Sugarless (for [now]) Python-like slicing (`A[a:b, c:]`) via the `slice` method:
  `A.slice((a..b, c..))`.
- Several flavors of iteration: over rows, over columns, over elements, or in "stripes".

[GEMM]: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
[MCOP]: https://en.wikipedia.org/wiki/Matrix_chain_multiplication
[now]: #improving-operator-sugar

## [API documentation]

[API documentation]: http://japaric.github.io/linalg.rs/linalg/

## [Examples]

[Examples]: https://github.com/japaric/linalg_examples

## C dependencies

The BLAS and LAPACK libraries must be available. In Ubuntu, you can install the reference
implementations with the following commands:

``` ignore
$ sudo apt-get install libblas-dev
$ sudo apt-get install liblapack-dev
```

Though an optimized BLAS package like [OpenBLAS] is highly recommend.

[OpenBLAS]: https://github.com/xianyi/OpenBLAS

## Improving operator sugar

If you saw the [quick reference] then you know that this library is not as nice to use as Numpy or
Octave due to the lack of operator sugar.

[quick reference]: http://japaric.github.io/linalg.rs/linalg/#quick-reference

Solving this issue requires changes in the compiler, here's a list with a brief description of each
change and the sugar it enables:

- Optional parenthesis when using tuples with the indexing notation. In other words `A[i, j]`
becomes the same as `A[(i, j)]`, this lets us match Python's indexing syntax.

``` rust
let x = &A[i, j];  // === Index::index(&A, (i, j));

let y = B[i, j];  // === *Index::index(&B, (i, j));
```

- User defined unsized types. This is big change, I've a WIP RFC [here]. This change improves
several things (check the RFC), but the syntatic improvement is that we'll be able to closely
match Python's slicing syntax:

``` rust
// &Row, &mut Col, &SubMat are fat pointers like &[T]

let row: &Row = &A[i, ..];  // === Index::index(&A, (i, ..));

let col: &mut Col = &mut A[.., j];  // === IndexMut::index_mut(&mut A, (.., j));

let submat: &SubMat = &A[a..b, c..d];  // === Index::index(&A, (a..b, c..d));
```

[here]: https://github.com/japaric/rfcs/blob/unsized/text/0000-unsized.md

- `[Op]Assign` traits. The RFC's [here]. This would let use augmented assignment sugar:

``` rust
A[i, ..] += &B[j, ..];  // === AddAssign::add_assign(&mut A[i, ..], &B[j, ..]);

A[a..b, c..d] *= 2;  // === MulAssign::mul_assign(&mut A[a..b, c..d], 2);
```

[here]: https://github.com/rust-lang/rfcs/pull/953

- `IndexSet`. There's a [postponed issue] for this. This would enable sugar for setting/copying
sub-matrices:

``` rust
A[1, ..] = 0;  // === IndexSet::index_set(&mut A[1, ..], 0);

A[..2, ..] = &B[2.., ..];  // === IndexSet::index_set(&mut A[..2, ..], &B[2.., ..]);
```

[postponed issue]: https://github.com/rust-lang/rfcs/issues/997

## Contributing

### Reporting bugs, missing operations and feature requests

The best way to contribute to this library is by using it and reporting any problem you
encounter.

If you hit a `debug_assert!`, that's a bug. Please open an issue in the [issue tracker]. Note that
`assert!`s are normal and indicate a programmer errors like index out of bounds or mismatched
dimensions.

[issue tracker]: https://github.com/japaric/linalg.rs/issues

If you get a wrong result from an operation, like `A * A.inv()` not being equal to the identity
matrix, that's a bug too.

If some operation doesn't compile and you think it should (but read the [Notes about operators]
first) that's a "missing operation" and should be reported.

[Notes about operators]: http://japaric.github.io/linalg.rs/linalg/#notes-about-operators

If you need some feature from the LAPACK/BLAS libraries that's not currently exposed, open a
feature request in the issue tracker.

### Unit tests

Ideally all the core functionality should be tested, but the current test suite is not that
extensive. There's a list of missing unit tests [here], help would be appreciated.

[here]: /TODO.test

### Examples

If you ported some (small) program from Numpy/Octave to `linalg`, consider submitting it as a
example.

### Benchmarks

If you have done any measurements between `linalg` and other linear algebra libraries (in any
language), Let me know how `linalg` performs, specially if measurements indicate that `linalg`
is slower.

## Similar/related work

- [algebloat]: Linear algebra library based on expression templates
- [nalgebra]: Contains a generic implementation of dense matrices
- [rblas]: Bindings to BLAS, and a dense matrix representation

[algebloat]: https://github.com/SiegeLord/RustAlgebloat
[nalgebra]: https://github.com/sebcrozet/nalgebra
[rblas]: https://github.com/mikkyang/rust-blas

## License

linalg.rs is dual licensed under the Apache 2.0 license and the MIT license.

See LICENSE-APACHE and LICENSE-MIT for more details.
