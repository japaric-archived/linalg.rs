//! A collection of the most used structs and traits, meant to be glob imported

pub use {Col, Mat, Row};
pub use complex::{Complex, c64, c128};
pub use traits::{
    AddAssign, At, AtMut, Collection, Iter, IterMut, Matrix, MatrixCol, MatrixColMut, MatrixCols,
    MatrixDiag, MatrixDiagMut, MatrixMutCols, MatrixMutRows, MatrixRow, MatrixRowMut, MatrixRows,
    MulAssign, Slice, SliceMut, SubAssign, ToOwned, Transpose,
};
