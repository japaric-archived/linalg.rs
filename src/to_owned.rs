use blas::{ToBlasInt, Vector, mod};
use strided;
use traits::{Collection, Matrix, MatrixCols, MatrixRows, ToOwned};
use {Col, Mat, MutView, Row, Trans, View};

fn to<T, V>(v: &V) -> Box<[T]> where V: Vector<T>, T: blas::Copy {
    let n = Collection::len(v);

    if n == 0 { return box [] }

    let copy = blas::Copy::copy(None::<T>);
    let x = v.as_ptr();
    let incx = v.stride();
    let mut data = Vec::with_capacity(n);
    let y = data.as_mut_ptr();
    let incy = 1;

    unsafe {
        copy(&n.to_blasint(), x, &incx, y, &incy);
        data.set_len(n)
    }

    data.into_boxed_slice()
}

macro_rules! impls {
    ($($ty:ty),+) => {$(
        impl<'a, T> ToOwned<Col<Box<[T]>>> for Col<$ty> where T: blas::Copy {
            fn to_owned(&self) -> Col<Box<[T]>> {
                Col(to(&self.0))
            }
        }

        impl<'a, T> ToOwned<Row<Box<[T]>>> for Row<$ty> where T: blas::Copy {
            fn to_owned(&self) -> Row<Box<[T]>> {
                Row(to(&self.0))
            }
        })+
    }
}

impls!(&'a [T], &'a mut [T], strided::Slice<'a, T>, strided::MutSlice<'a, T>)

impl<'a, T> ToOwned<Col<Box<[T]>>> for Col<Box<[T]>> where T: blas::Copy {
    fn to_owned(&self) -> Col<Box<[T]>> {
        Col(to(&self.0))
    }
}

impl<'a, T> ToOwned<Row<Box<[T]>>> for Row<Box<[T]>> where T: blas::Copy {
    fn to_owned(&self) -> Row<Box<[T]>> {
        Row(to(&self.0))
    }
}

impl<T> ToOwned<Mat<T>> for Mat<T> where T: blas::Copy {
    fn to_owned(&self) -> Mat<T> {
        Mat {
            data: to(&self.data),
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
