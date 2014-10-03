use std::fmt::{Formatter, Show, mod};
use traits::{Iter, MatrixRows};
use {Mat, MutView, Trans, View};

// XXX Sadly, I can't implement `Show` generically, so I'll repeat myself using macros

// TODO Precision, padding
macro_rules! fmt {
    () => {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
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

impl<T> Show for Mat<T> where T: Show { fmt!() }
impl<T> Show for Trans<Mat<T>> where T: Show { fmt!() }

macro_rules! impl_show {
    ($($ty:ty),+) => {$(
        impl<'a, T: Show> Show for $ty {
            fmt!()
        }
    )+}
}

impl_show!(MutView<'a, T>, Trans<MutView<'a, T>>, Trans<View<'a, T>>, View<'a, T>)
