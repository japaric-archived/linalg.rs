use onezero::One;

use {
    Col, ColVec, Mat, MutCol, MutDiag, MutRow, MutView, Row, RowVec, Scaled, ToBlasint, Trans,
    View,
};
use blas::axpy::Axpy;
use traits::{AddAssign, Matrix, MatrixCols, MatrixMutCols, MatrixMutRows, MatrixRows};

// vector += scalar
fn vs<T>(lhs: ::raw::strided::Slice<T>, rhs: T) where T: Axpy + One {
    let n = lhs.len;

    if n == 0 { return }

    let axpy = Axpy::axpy(None::<T>);
    let alpha = One::one();
    let x = &rhs;
    let incx = 0;
    let y = lhs.data;
    let incy = lhs.stride.to_blasint();

    unsafe { axpy(&n.to_blasint(), &alpha, x, &incx, y, &incy) }
}

// col
impl<T> AddAssign<T> for ColVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        self.as_mut_col().add_assign(rhs)
    }
}

impl<'a, T> AddAssign<T> for MutCol<'a, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        vs((self.0).0, rhs)
    }
}

// diag
impl<'a, T> AddAssign<T> for MutDiag<'a, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        vs((self.0).0, rhs)
    }
}

// mat
impl<T> AddAssign<T> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        self.unroll_mut().add_assign(rhs)
    }
}

impl<'a, T> AddAssign<T> for MutView<'a, T> where T: Axpy + Clone + One {
    fn add_assign(&mut self, rhs: T) {
        if self.nrows() < self.ncols() {
            for mut row in self.mut_rows() {
                row.add_assign(rhs.clone())
            }
        } else {
            for mut col in self.mut_cols() {
                col.add_assign(rhs.clone())
            }
        }
    }
}

impl<T> AddAssign<T> for Trans<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        self.0.add_assign(rhs)
    }
}

impl<'a, T> AddAssign<T> for Trans<MutView<'a, T>> where T: Axpy + Clone + One {
    fn add_assign(&mut self, rhs: T) {
        self.0.add_assign(rhs)
    }
}

// row
impl<'a, T> AddAssign<T> for MutRow<'a, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        vs((self.0).0, rhs)
    }
}

impl<T> AddAssign<T> for RowVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        self.as_mut_row().add_assign(rhs)
    }
}

// vector += vector
fn vv<T>(lhs: ::raw::strided::Slice<T>, alpha: T, rhs: ::raw::strided::Slice<T>) where T: Axpy {
    assert_eq!(lhs.len, rhs.len);

    let n = lhs.len;

    if n == 0 { return }

    let axpy = Axpy::axpy(None::<T>);
    let n = n.to_blasint();
    let x = rhs.data;
    let incx = rhs.stride.to_blasint();
    let y = lhs.data;
    let incy = lhs.stride.to_blasint();

    unsafe { axpy(&n, &alpha, x, &incx, y, &incy) }
}

// col
impl<'a, T> AddAssign<Col<'a, T>> for ColVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Col<T>) {
        self.as_mut_col().add_assign(rhs)
    }
}

impl<'a, T> AddAssign<&'a ColVec<T>> for ColVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &ColVec<T>) {
        self.as_mut_col().add_assign(rhs.as_col())
    }
}

impl<'a, 'b, T> AddAssign<&'a MutCol<'b, T>> for ColVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &MutCol<T>) {
        self.as_mut_col().add_assign(rhs.as_col())
    }
}

impl<'a, T> AddAssign<Scaled<T, Col<'a, T>>> for ColVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Scaled<T, Col<T>>) {
        self.as_mut_col().add_assign(rhs)
    }
}

impl<'a, 'b, T> AddAssign<Col<'a, T>> for MutCol<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Col<T>) {
        vv((self.0).0, One::one(), (rhs.0).0)
    }
}

impl<'a, 'b, T> AddAssign<&'a ColVec<T>> for MutCol<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &ColVec<T>) {
        self.add_assign(rhs.as_col())
    }
}

impl<'a, 'b, 'c, T> AddAssign<&'a MutCol<'b, T>> for MutCol<'c, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &MutCol<T>) {
        self.add_assign(rhs.as_col())
    }
}

impl<'a, 'b, T> AddAssign<Scaled<T, Col<'a, T>>> for MutCol<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Scaled<T, Col<T>>) {
        vv((self.0).0, rhs.0, ((rhs.1).0).0)
    }
}

// matrix += matrix
macro_rules! mm {
    ($lhs:ty) => {
        fn add_assign(&mut self, rhs: $lhs) {
            assert_eq!(self.size(), rhs.size());

            if self.nrows() < self.ncols() {
                for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                    lhs.add_assign(rhs)
                }
            } else {
                for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                    lhs.add_assign(rhs)
                }
            }
        }
    }
}

// mat
// XXX (nit) double "equal size" assertion
impl<'a, T> AddAssign<&'a Mat<T>> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Mat<T>) {
        assert_eq!(self.size(), rhs.size());

        self.unroll_mut().add_assign(rhs.unroll())
    }
}

impl<'a, 'b, T> AddAssign<&'a MutView<'b, T>> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &MutView<T>) {
        self.as_mut_view().add_assign(rhs.as_view())
    }
}

impl<'a, T> AddAssign<Scaled<T, Trans<View<'a, T>>>> for Mat<T> where T: Axpy + Clone + One {
    fn add_assign(&mut self, rhs: Scaled<T, Trans<View<T>>>) {
        self.as_mut_view().add_assign(rhs)
    }
}

impl<'a, T> AddAssign<Scaled<T, View<'a, T>>> for Mat<T> where T: Axpy + Clone + One {
    fn add_assign(&mut self, rhs: Scaled<T, View<T>>) {
        self.as_mut_view().add_assign(rhs)
    }
}

impl<'a, T> AddAssign<&'a Trans<Mat<T>>> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.as_mut_view().add_assign(Trans(rhs.0.as_view()))
    }
}

impl<'a, 'b, T> AddAssign<&'a Trans<MutView<'b, T>>> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.as_mut_view().add_assign(Trans(rhs.0.as_view()))
    }
}

impl<'a, 'b, T> AddAssign<Trans<View<'b, T>>> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Trans<View<T>>) {
        self.as_mut_view().add_assign(rhs)
    }
}

impl<'a, 'b, T> AddAssign<View<'a, T>> for Mat<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: View<T>) {
        self.as_mut_view().add_assign(rhs)
    }
}

impl<'a, 'b, T> AddAssign<&'a Mat<T>> for MutView<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Mat<T>) {
        self.add_assign(rhs.as_view())
    }
}

impl<'a, 'b, 'c, T> AddAssign<&'a MutView<'b, T>> for MutView<'c, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &MutView<T>) {
        self.add_assign(rhs.as_view())
    }
}

impl<'a, 'b, T> AddAssign<Scaled<T, Trans<View<'a, T>>>> for MutView<'b, T> where
    T: Axpy + Clone + One,
{
    mm!(Scaled<T, Trans<View<T>>>);
}

impl<'a, 'b, T> AddAssign<Scaled<T, View<'a, T>>> for MutView<'b, T> where T: Axpy + Clone + One {
    mm!(Scaled<T, View<T>>);
}

impl<'a, 'b, T> AddAssign<&'a Trans<Mat<T>>> for MutView<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.add_assign(Trans(rhs.0.as_view()))
    }
}

impl<'a, 'b, 'c, T> AddAssign<&'a Trans<MutView<'b, T>>> for MutView<'c, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.add_assign(Trans(rhs.0.as_view()))
    }
}

impl<'a, 'b, T> AddAssign<Trans<View<'a, T>>> for MutView<'b, T> where T: Axpy + One {
    mm!(Trans<View<T>>);
}

impl<'a, 'b, T> AddAssign<View<'a, T>> for MutView<'b, T> where T: Axpy + One {
    mm!(View<T>);
}

impl<'a, T> AddAssign<&'a Mat<T>> for Trans<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Mat<T>) {
        Trans(self.0.as_mut_view()).add_assign(rhs.as_view())
    }
}

impl<'a, 'b, T> AddAssign<&'a MutView<'b, T>> for Trans<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &MutView<T>) {
        Trans(self.0.as_mut_view()).add_assign(rhs.as_view())
    }
}

impl<'a, T> AddAssign<Scaled<T, Trans<View<'a, T>>>> for Trans<Mat<T>> where
    T: Axpy + Clone + One,
{
    fn add_assign(&mut self, rhs: Scaled<T, Trans<View<T>>>) {
        self.0.as_mut_view().add_assign(Scaled(rhs.0, (rhs.1).0))
    }
}

impl<'a, T> AddAssign<Scaled<T, View<'a, T>>> for Trans<Mat<T>> where T: Axpy + Clone + One {
    fn add_assign(&mut self, rhs: Scaled<T, View<T>>) {
        Trans(self.0.as_mut_view()).add_assign(rhs)
    }
}

impl<'a, T> AddAssign<&'a Trans<Mat<T>>> for Trans<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.0.add_assign(rhs.0.as_view())
    }
}

impl<'a, 'b, T> AddAssign<&'a Trans<MutView<'b, T>>> for Trans<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.0.add_assign(rhs.0.as_view())
    }
}

impl<'a, T> AddAssign<Trans<View<'a, T>>> for Trans<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Trans<View<T>>) {
        self.0.add_assign(rhs.0)
    }
}

impl<'a, T> AddAssign<View<'a, T>> for Trans<Mat<T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: View<T>) {
        Trans(self.0.as_mut_view()).add_assign(rhs)
    }
}

impl<'a, 'b, T> AddAssign<&'a Mat<T>> for Trans<MutView<'b, T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Mat<T>) {
        self.add_assign(rhs.as_view())
    }
}

impl<'a, 'b, 'c, T> AddAssign<&'a MutView<'b, T>> for Trans<MutView<'c, T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &MutView<T>) {
        self.add_assign(rhs.as_view())
    }
}

impl<'a, 'b, T> AddAssign<Scaled<T, Trans<View<'a, T>>>> for Trans<MutView<'b, T>> where
    T: Axpy + Clone + One,
{
    fn add_assign(&mut self, rhs: Scaled<T, Trans<View<T>>>) {
        self.0.add_assign(Scaled(rhs.0, (rhs.1).0))
    }
}

impl<'a, 'b, T> AddAssign<Scaled<T, View<'a, T>>> for Trans<MutView<'b, T>> where
    T: Axpy + Clone + One,
{
    mm!(Scaled<T, View<T>>);
}

impl<'a, 'b, T> AddAssign<&'a Trans<Mat<T>>> for Trans<MutView<'b, T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.0.add_assign(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> AddAssign<&'a Trans<MutView<'b, T>>> for Trans<MutView<'c, T>> where
    T: Axpy + One,
{
    fn add_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.0.add_assign(rhs.0.as_view())
    }
}

impl<'a, 'b, T> AddAssign<Trans<View<'a, T>>> for Trans<MutView<'b, T>> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Trans<View<T>>) {
        self.0.add_assign(rhs.0)
    }
}

impl<'a, 'b, T> AddAssign<View<'a, T>> for Trans<MutView<'b, T>> where T: Axpy + One {
    mm!(View<T>);
}

// row
impl<'a, 'b, 'c, T> AddAssign<&'a MutRow<'b, T>> for MutRow<'c, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &MutRow<T>) {
        self.add_assign(rhs.as_row())
    }
}

impl<'a, 'b, T> AddAssign<Row<'a, T>> for MutRow<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Row<T>) {
        vv((self.0).0, One::one(), (rhs.0).0)
    }
}

impl<'a, 'b, T> AddAssign<&'a RowVec<T>> for MutRow<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &RowVec<T>) {
        self.add_assign(rhs.as_row())
    }
}

impl<'a, 'b, T> AddAssign<Scaled<T, Row<'a, T>>> for MutRow<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Scaled<T, Row<T>>) {
        vv((self.0).0, rhs.0, ((rhs.1).0).0)
    }
}

impl<'a, 'b, T> AddAssign<&'a MutRow<'b, T>> for RowVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &MutRow<T>) {
        self.as_mut_row().add_assign(rhs.as_row())
    }
}

impl<'a, T> AddAssign<Row<'a, T>> for RowVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Row<T>) {
        self.as_mut_row().add_assign(rhs)
    }
}

impl<'a, T> AddAssign<&'a RowVec<T>> for RowVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &RowVec<T>) {
        self.as_mut_row().add_assign(rhs.as_row())
    }
}

impl<'a, T> AddAssign<Scaled<T, Row<'a, T>>> for RowVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Scaled<T, Row<T>>) {
        self.as_mut_row().add_assign(rhs)
    }
}
