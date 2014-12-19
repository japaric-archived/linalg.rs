use onezero::One;

use blas::{Axpy, MutVector, ToBlasInt, Vector};
use strided;
use traits::{
    AddAssign, Collection, Matrix, MatrixCols, MatrixMutCols, MatrixMutRows, MatrixRows
};
use {Col, Diag, Mat, MutView, Row, Trans, View};

macro_rules! add_assign {
    ($($lhs:ty $rhs:ty),+,) => {$(
        impl<T, L> AddAssign<T> for $lhs where
            T: Axpy + One,
            L: MutVector<T>,
        {
            fn add_assign(&mut self, rhs: &T) {
                vs(&mut self.0, rhs)
            }
        }

        impl<T, L, R> AddAssign<$rhs> for $lhs where
            T: Axpy + One,
            L: MutVector<T>,
            R: Vector<T>,
        {
            fn add_assign(&mut self, rhs: &$rhs) {
                vv(&mut self.0, &rhs.0)
            }
        })+
    }
}

add_assign! {
    Col<L> Col<R>,
    Row<L> Row<R>,
}

impl<'a, T> AddAssign<T> for Diag<strided::MutSlice<'a, T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        vs(&mut self.0, rhs)
    }
}

impl<T> AddAssign<T> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        let n = Collection::len(&self.data);

        if n == 0 { return }

        let axpy = Axpy::axpy(None::<T>);
        let n = n.to_blasint();
        let alpha = One::one();
        let x = rhs;
        let incx = 0;
        let y = self.data.as_mut_ptr();
        let incy = 1;

        unsafe { axpy(&n, &alpha, x, &incx, y, &incy) }
    }
}

impl<T> AddAssign<T> for Trans<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        self.0.add_assign(rhs)
    }
}

impl<T> AddAssign<Mat<T>> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Mat<T>) {
        assert_eq!(self.size(), rhs.size());

        let n = Collection::len(&self.data);

        if n == 0 { return }

        let axpy = Axpy::axpy(None::<T>);
        let n = n.to_blasint();
        let alpha = One::one();
        let x = rhs.data.as_ptr();
        let incx = 1;
        let y = self.data.as_mut_ptr();
        let incy = 1;

        unsafe { axpy(&n, &alpha, x, &incx, y, &incy) }
    }
}

impl<T> AddAssign<Trans<Mat<T>>> for Trans<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.0.add_assign(&rhs.0)
    }
}

impl<T> AddAssign<Trans<Mat<T>>> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Trans<Mat<T>>) {
        assert_eq!(self.size(), rhs.size());

        if self.nrows() < self.ncols() {
            for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                lhs.add_assign(&rhs)
            }
        } else {
            for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                lhs.add_assign(&rhs)
            }
        }
    }
}

impl<T> AddAssign<Mat<T>> for Trans<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Mat<T>) {
        assert_eq!(self.size(), rhs.size());

        if self.nrows() < self.ncols() {
            for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                lhs.add_assign(&rhs)
            }
        } else {
            for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                lhs.add_assign(&rhs)
            }
        }
    }
}

impl<'a, T> AddAssign<T> for MutView<'a, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        if self.nrows() < self.ncols() {
            for mut row in self.mut_rows() {
                row.add_assign(rhs)
            }
        } else {
            for mut col in self.mut_cols() {
                col.add_assign(rhs)
            }
        }
    }
}

impl<'a, T> AddAssign<T> for Trans<MutView<'a, T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        self.0.add_assign(rhs)
    }
}

macro_rules! impls {
    ($($lhs:ty $rhs:ty),+,) => {$(
        impl<'a, T> AddAssign<$rhs> for $lhs where T: Axpy + One {
            fn add_assign(&mut self, rhs: &$rhs) {
                assert_eq!(self.size(), rhs.size());

                if self.nrows() < self.ncols() {
                    for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                        lhs.add_assign(&rhs)
                    }
                } else {
                    for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                        lhs.add_assign(&rhs)
                    }
                }
            }
        })+
     }
}

impls! {
    MutView<'a, T> Mat<T>,
    MutView<'a, T> Trans<Mat<T>>,
    Trans<MutView<'a, T>> Mat<T>,
    Trans<MutView<'a, T>> Trans<Mat<T>>,
}

macro_rules! view {
    ($($ty:ty),+) => {$(
        impl<'a, T> AddAssign<$ty> for Mat<T> where T: Axpy + One {
            fn add_assign(&mut self, rhs: &$ty) {
                assert_eq!(self.size(), rhs.size());

                if self.nrows() < self.ncols() {
                    for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                        lhs.add_assign(&rhs)
                    }
                } else {
                    for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                        lhs.add_assign(&rhs)
                    }
                }
            }
        }

        impl<'a, T> AddAssign<Trans<$ty>> for Mat<T> where T: Axpy + One {
            fn add_assign(&mut self, rhs: &Trans<$ty>) {
                assert_eq!(self.size(), rhs.size());

                if self.nrows() < self.ncols() {
                    for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                        lhs.add_assign(&rhs)
                    }
                } else {
                    for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                        lhs.add_assign(&rhs)
                    }
                }
            }
        }

        impl<'a, T> AddAssign<$ty> for Trans<Mat<T>> where T: Axpy + One {
            fn add_assign(&mut self, rhs: &$ty) {
                assert_eq!(self.size(), rhs.size());

                if self.nrows() < self.ncols() {
                    for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                        lhs.add_assign(&rhs)
                    }
                } else {
                    for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                        lhs.add_assign(&rhs)
                    }
                }
            }
        }

        impl<'a, T> AddAssign<Trans<$ty>> for Trans<Mat<T>> where T: Axpy + One {
            fn add_assign(&mut self, rhs: &Trans<$ty>) {
                assert_eq!(self.size(), rhs.size());

                if self.nrows() < self.ncols() {
                    for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                        lhs.add_assign(&rhs)
                    }
                } else {
                    for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                        lhs.add_assign(&rhs)
                    }
                }
            }
        }

        impl<'a, 'b, T> AddAssign<$ty> for MutView<'b, T> where T: Axpy + One {
            fn add_assign(&mut self, rhs: &$ty) {
                assert_eq!(self.size(), rhs.size());

                if self.nrows() < self.ncols() {
                    for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                        lhs.add_assign(&rhs)
                    }
                } else {
                    for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                        lhs.add_assign(&rhs)
                    }
                }
            }
        }

        impl<'a, 'b, T> AddAssign<Trans<$ty>> for MutView<'b, T> where T: Axpy + One {
            fn add_assign(&mut self, rhs: &Trans<$ty>) {
                assert_eq!(self.size(), rhs.size());

                if self.nrows() < self.ncols() {
                    for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                        lhs.add_assign(&rhs)
                    }
                } else {
                    for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                        lhs.add_assign(&rhs)
                    }
                }
            }
        }

        impl<'a, 'b, T> AddAssign<$ty> for Trans<MutView<'b, T>> where T: Axpy + One {
            fn add_assign(&mut self, rhs: &$ty) {
                assert_eq!(self.size(), rhs.size());

                if self.nrows() < self.ncols() {
                    for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                        lhs.add_assign(&rhs)
                    }
                } else {
                    for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                        lhs.add_assign(&rhs)
                    }
                }
            }
        }

        impl<'a, 'b, T> AddAssign<Trans<$ty>> for Trans<MutView<'b, T>> where T: Axpy + One {
            fn add_assign(&mut self, rhs: &Trans<$ty>) {
                assert_eq!(self.size(), rhs.size());

                if self.nrows() < self.ncols() {
                    for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                        lhs.add_assign(&rhs)
                    }
                } else {
                    for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                        lhs.add_assign(&rhs)
                    }
                }
            }
        })+
    }
}

view!(View<'a, T>, MutView<'a, T>);

fn vs<T, V: MutVector<T>>(lhs: &mut V, rhs: &T) where T: Axpy + One {
    let n = Collection::len(lhs);

    if n == 0 { return }

    let axpy = Axpy::axpy(None::<T>);
    let n = n.to_blasint();
    let alpha = One::one();
    let x = rhs;
    let incx = 0;
    let y = lhs.as_mut_ptr();
    let incy = lhs.stride();

    unsafe { axpy(&n, &alpha, x, &incx, y, &incy) }
}

fn vv<T, L: MutVector<T>, R: Vector<T>>(lhs: &mut L, rhs: &R) where T: Axpy + One {
    assert_eq!(Collection::len(lhs), Collection::len(rhs));

    let n = Collection::len(lhs);

    if n == 0 { return }

    let axpy = Axpy::axpy(None::<T>);
    let n = n.to_blasint();
    let alpha = One::one();
    let x = rhs.as_ptr();
    let incx = rhs.stride();
    let y = lhs.as_mut_ptr();
    let incy = lhs.stride();

    unsafe { axpy(&n, &alpha, x, &incx, y, &incy) }
}
