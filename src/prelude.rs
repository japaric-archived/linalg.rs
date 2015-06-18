//! The "prelude", a collection of the most used traits and structs.
//!
//! This module is meant to be manually glob imported.

pub use {Col, Row};

pub use traits::Matrix as __linalg_0;
pub use traits::Norm as __linalg_1;
pub use traits::Transpose as __linalg_2;
