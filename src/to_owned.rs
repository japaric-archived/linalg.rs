use blas::{ToBlasInt, Vector, mod};
use traits::{Collection, Matrix, MatrixCols, MatrixRows, ToOwned};
use {Col, Mat, MutView, Row, Trans, View};

impl<T, V> ToOwned<Box<[T]>> for V where T: blas::Copy, V: Vector<T> {
    fn to_owned(&self) -> Box<[T]> {
        let n = Collection::len(self);

        if n == 0 { return box [] }

        let copy = blas::Copy::copy(None::<T>);
        let x = self.as_ptr();
        let incx = self.stride();
        let mut data = Vec::with_capacity(n);
        let y = data.as_mut_ptr();
        let incy = 1;

        unsafe {
            copy(&n.to_blasint(), x, &incx, y, &incy);
            data.set_len(n)
        }

        data.into_boxed_slice()
    }
}

impl<T, V> ToOwned<Col<Box<[T]>>> for Col<V> where T: blas::Copy, V: Vector<T> {
    fn to_owned(&self) -> Col<Box<[T]>> {
        Col(self.0.to_owned())
    }
}

impl<T, V> ToOwned<Row<Box<[T]>>> for Row<V> where T: blas::Copy, V: Vector<T> {
    fn to_owned(&self) -> Row<Box<[T]>> {
        Row(self.0.to_owned())
    }
}

impl<T> ToOwned<Mat<T>> for Mat<T> where T: blas::Copy {
    fn to_owned(&self) -> Mat<T> {
        Mat {
            data: self.data.to_owned(),
            size: self.size,
        }
    }
}

macro_rules! to_owned {
    () => {
        fn to_owned(&self) -> Mat<T> {
            // XXX Should this use `checked_mul`?
            let n = self.nrows() * self.ncols();

            if n == 0 {
                return Mat {
                    data: box [],
                    size: self.size(),
                }
            }

            let mut data = Vec::with_capacity(n);

            let copy = blas::Copy::copy(None::<T>);

            if self.nrows() < self.ncols() {
                let n = self.ncols().to_blasint();
                let incy = self.nrows().to_blasint();

                for (i, row) in self.rows().enumerate() {
                    let x = row.0.as_ptr();
                    let incx = row.0.stride();
                    let y = unsafe { data.as_mut_ptr().offset(i as int) };

                    unsafe { copy(&n, x, &incx, y, &incy) }
                }
            } else {
                let n = self.nrows().to_blasint();
                let incy = 1;

                for (i, col) in self.cols().enumerate() {
                    let x = col.0.as_ptr();
                    let incx = col.0.stride();
                    let y = unsafe { data.as_mut_ptr().offset((i * self.nrows()) as int) };

                    unsafe { copy(&n, x, &incx, y, &incy) }
                }
            }

            unsafe { data.set_len(n) }

            Mat {
                data: data.into_boxed_slice(),
                size: self.size(),
            }
        }
    }
}

impl<T> ToOwned<Mat<T>> for Trans<Mat<T>> where T: blas::Copy {
    to_owned!()
}

impl<'a, T> ToOwned<Mat<T>> for View<'a, T> where T: blas::Copy {
    to_owned!()
}

impl<'a, T> ToOwned<Mat<T>> for MutView<'a, T> where T: blas::Copy {
    to_owned!()
}

impl<'a, T> ToOwned<Mat<T>> for Trans<View<'a, T>> where T: blas::Copy {
    to_owned!()
}

impl<'a, T> ToOwned<Mat<T>> for Trans<MutView<'a, T>> where T: blas::Copy {
    to_owned!()
}
