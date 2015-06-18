**WARNING** Right now this crate only works in the (unofficial) [edge] channel, which contains
features that are still in the RFC phase.

[edge]: https://github.com/japaric/rusty-edge

# `linalg.rs`

A linear algebra library for numerical/scientific computing.

## Features

- Dense matrices and row/column vectors with full arithmetic support [1]
- Expression simplifier: `alpha * A + beta * B + C` is evaluated as a single GEMM/GEMV call
- Python/NumPy-like indexing/slicing syntax: `A[i, j]`, `B[.., j]`, `C[a..b, c..d]`
- Several flavors of iterators: element-wise, column-by-column, row-by-row, or in "stripes" [2]
- Support for both column major and row major memory order.
- Support for smart pointers: `Cow<Mat>`, `Rc<Mat>` [3]
- Zero cost transpose

[1] Right now I intend to cover all the arithmetic operations that directly map to BLAS calls

[2] I'm particularly fond of [this code] that arranges several images in a [matrix].

[3] `linalg` supports the `Rc` pointer defined in the [rc] crate, and not the one in the `std`
crate, because the latter doesn't (yet?) provide a method to convert from `Box<Unsized>` to
`Rc<Unsized>`.

[this code]: https://github.com/japaric/linalg.rs/blob/ng/src/nn/images.rs#L128-134
[matrix]: https://github.com/japaric/linalg.rs/blob/ng/src/nn/training_set.png
[rc]: https://github.com/japaric/rc.rs

## Planned features

- Expression templates: Create custom kernels at compile time to cover operations that BLAS doesn't
  cover. See early experiments [here].
- API for matrix decompositions: rustic interface to LAPACK routines
- Transparent CUDA (cuBLAS) acceleration. The bottleneck of GPU computing are the host <-> GPU
  data transfers, so I'll add a new type `cuda::Mat` that manages GPU memory, and arithmetic
  operations will only be allowed between matrices stored in the GPU, so no implicit host <-> GPU
  data transfers. Sugar will be provided for host <-> GPU transfers: `gpu_mat[..] = host_mat`.

[here]: https://github.com/japaric/et.rs

## [API docs]

[API docs]: http://japaric.github.io/linalg.rs/linalg/

**NOTE** Unsized types like `Mat<T, O>` are (wrongly) rendered as empty enums `enum Mat<T, O> {}` by
rustdoc because I haven't got around to add proper rendering support.

---

## Fixed issues

Closes #22 support row major order

Closes #56 internally use U31 newtype instead of i32

Closes #62 support conversions from &Col/&Row to &Mat

Closes #63 replace Transposed newtype with phantom type that stores the "order" (column/row)

Closes #64 distinguish strided from contiguous row/column vectors

Closes #66 can't send mutable block of a matrix to scoped threads

Closes #68 use the One/Zero traits provided by the std crate

Closes #73 implement `&mut Mat + &Mat -> &mut Mat`

Closes #75 replace the `Set` trait with `IndexAssign`
