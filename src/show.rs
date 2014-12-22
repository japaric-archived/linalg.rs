use std::fmt;

use {Col, ColVec, Diag, Mat, MutCol, MutDiag, MutRow, MutView, Row, RowVec, Trans, View};
use traits::{Iter, MatrixRows};

// TODO Precision, padding
macro_rules! fmt {
    () => {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut is_first_row = true;
            for row in self.rows() {
                if is_first_row {
                    is_first_row = false;
                } else {
                    try!(writeln!(f, ""));
                }

                try!(write!(f, "["));

                let mut is_first_col = true;
                for e in row.iter() {
                    if is_first_col {
                        is_first_col = false;
                    } else {
                        try!(write!(f, ", "));
                    }
                    try!(write!(f, "{}", e));
                }

                try!(write!(f, "]"))
            }

            Ok(())
        }
    }
}

impl<T> fmt::Show for Mat<T> where T: fmt::Show { fmt!(); }
impl<T> fmt::Show for Trans<Mat<T>> where T: fmt::Show { fmt!(); }

macro_rules! mat_impls {
    ($($ty:ty),+) => {
        $(
            impl<'a, T> fmt::Show for $ty where T: fmt::Show {
                fmt!();
            }
        )+
    }
}

mat_impls!(MutView<'a, T>, Trans<MutView<'a, T>>, Trans<View<'a, T>>, View<'a, T>);

impl<'a, T> fmt::Show for ColVec<T> where T: fmt::Show {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Col({})", self.0)
    }
}

impl<'a, T> fmt::Show for RowVec<T> where T: fmt::Show {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Row({})", self.0)
    }
}

macro_rules! impls {
    ($($ty:ty -> $str:expr),+,) => {
        $(
            impl<'a, T> fmt::Show for $ty where T: fmt::Show {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, concat!($str, "({})"), self.0)
                }
            }
        )+
    }
}

impls! {
    Col<'a, T> -> "Col",
    Diag<'a, T> -> "Diag",
    MutCol<'a, T> -> "Col",
    MutDiag<'a, T> -> "Diag",
    MutRow<'a, T> -> "Row",
    Row<'a, T> -> "Row",
}
