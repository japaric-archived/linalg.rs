//! The "prelude", a collection of the most used traits and structs.
//!
//! This module is meant to be manually glob imported.

pub use ColVec;
pub use Mat;
pub use RowVec;

pub use assign::AddAssign as __linalg_0;
pub use assign::DivAssign as __linalg_1;
pub use assign::MulAssign as __linalg_2;
pub use assign::SubAssign as __linalg_3;

pub use traits::Eval as __linalg_4;
pub use traits::HSplit as __linalg_5;
pub use traits::HSplitMut as __linalg_6;
pub use traits::Iter as __linalg_7;
pub use traits::IterMut as __linalg_8;
pub use traits::Matrix as __linalg_9;
pub use traits::MatrixCol as __linalg_10;
pub use traits::MatrixColMut as __linalg_11;
pub use traits::MatrixCols as __linalg_12;
pub use traits::MatrixColsMut as __linalg_13;
pub use traits::MatrixDiag as __linalg_14;
pub use traits::MatrixDiagMut as __linalg_15;
pub use traits::MatrixHStripes as __linalg_16;
pub use traits::MatrixHStripesMut as __linalg_17;
pub use traits::MatrixInverse as __linalg_18;
pub use traits::MatrixRow as __linalg_19;
pub use traits::MatrixRowMut as __linalg_20;
pub use traits::MatrixRows as __linalg_21;
pub use traits::MatrixRowsMut as __linalg_22;
pub use traits::MatrixVStripes as __linalg_23;
pub use traits::MatrixVStripesMut as __linalg_24;
pub use traits::Set as __linalg_25;
pub use traits::Slice as __linalg_26;
pub use traits::SliceMut as __linalg_27;
pub use traits::Transpose as __linalg_28;
pub use traits::VSplit as __linalg_29;
pub use traits::VSplitMut as __linalg_30;
