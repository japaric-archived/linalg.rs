//! A collection of the most used structs and traits, meant to be glob imported

pub use assign::{AddAssign, MulAssign, SubAssign};
pub use complex::{Complex, c64, c128};
pub use traits::{
    At, AtMut, Iter, IterMut, Matrix, MatrixCol, MatrixColMut, MatrixCols, MatrixDiag,
    MatrixDiagMut, MatrixMutCols, MatrixMutRows, MatrixRow, MatrixRowMut, MatrixRows, Slice,
    SliceMut, ToOwned, Transpose,
};
pub use {ColVec, Mat, RowVec};
