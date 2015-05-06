use std::fmt;

use traits::{MatrixRows, Slice};
use {Col, ColMut, ColVec, Diag, DiagMut, Mat, Row, RowMut, RowVec, Transposed, SubMat, SubMatMut};

macro_rules! fmt {
    () => {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            use traits::MatrixRows;

            let mut is_first = true;
            for row in self.rows() {
                if is_first {
                    is_first = false;
                } else {
                    try!(f.write_str("\n"))
                }

                try!(row.0.fmt(f))
            }

            Ok(())
        }
    };
}

// Combinations:
//
// - Col, ColMut, ColVec
// - Diag, DiagMut
// - Mat, SubMat, SubMatMut
// - Row, RowMut, RowVec
// - Transposed<Mat>, Transposed<SubMat>, Transposed<SubMatMut>
//
// -> 14 impls

// 5 impls
// Core implementations
impl<'a, T> fmt::Debug for Transposed<SubMat<'a, T>> where T: fmt::Debug {
    fmt!();
}

impl<'a, T> fmt::Debug for SubMat<'a, T> where T: fmt::Debug {
    fmt!();
}

impl<'a, T> fmt::Debug for Col<'a, T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Col({:?})", self.0)
    }
}

impl<'a, T> fmt::Debug for Diag<'a, T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Diag({:?})", self.0)
    }
}

impl<'a, T> fmt::Debug for Row<'a, T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Row({:?})", self.0)
    }
}

macro_rules! forward {
    ($($ty:ty),+,) => {
        $(
            impl<'a, T> fmt::Debug for $ty where T: fmt::Debug {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    self.slice(..).fmt(f)
                }
            }
         )+
    }
}

// 9 impls
forward! {
    ColMut<'a, T>,
    ColVec<T>,
    DiagMut<'a, T>,
    Mat<T>,
    RowMut<'a, T>,
    RowVec<T>,
    Transposed<Mat<T>>,
    Transposed<SubMatMut<'a, T>>,
    SubMatMut<'a, T>,
}
