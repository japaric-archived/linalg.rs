use blas::Copy;

use ops::{from, self};
use traits::{Matrix, MatrixCols, MatrixColsMut, MatrixRows, MatrixRowsMut, Slice};
use {ColMut, Col, Mat, RowMut, Row, Transposed, SubMat, SubMatMut};

// NOTE Core
impl<'a, T> From<SubMat<'a, T>> for Mat<T> where T: Copy {
    fn from(input: SubMat<T>) -> Mat<T> {
        unsafe {
            if let Some(slice) = input.as_slice() {
                Mat {
                    data: from::slice(slice),
                    ncols: input.ncols,
                    nrows: input.nrows,
                }
            } else {
                let mut m = Mat::uninitialized((input.nrows, input.ncols));

                if input.nrows < input.ncols {
                    for (Row(ref x), RowMut(Row(ref mut y))) in input.rows().zip(m.rows_mut()) {
                        ops::copy_strided(x, y)
                    }
                } else {
                    for (Col(ref x), ColMut(Col(ref mut y))) in input.cols().zip(m.cols_mut()) {
                        ops::copy_strided(x, y)
                    }
                }

                m
            }
        }
    }
}

// NOTE Core
impl<'a, T> From<Transposed<SubMat<'a, T>>> for Mat<T> where T: Copy {
    fn from(input: Transposed<SubMat<T>>) -> Mat<T> {
        unsafe {
            let mut m = Mat::uninitialized((input.0.ncols, input.0.nrows));

            if input.nrows() < input.ncols() {
                for (Row(ref x), RowMut(Row(ref mut y))) in input.rows().zip(m.rows_mut()) {
                    ops::copy_strided(x, y)
                }
            } else {
                for (Col(ref x), ColMut(Col(ref mut y))) in input.cols().zip(m.cols_mut()) {
                    ops::copy_strided(x, y)
                }
            }

            m
        }
    }
}

macro_rules! forward {
    ($($src:ty),+,) => {
        $(
            impl<'a, 'b, T> From<$src> for Mat<T> where T: Copy {
                fn from(input: $src) -> Mat<T> {
                    Mat::from(input.slice(..))
                }
            }
         )+
    }
}

forward! {
    &'a SubMatMut<'b, T>,
    &'a Transposed<Mat<T>>,
    &'a Transposed<SubMatMut<'b, T>>,
}
