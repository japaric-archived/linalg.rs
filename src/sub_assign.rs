use onezero::One;
use std::ops::Neg;

use {
    Col, ColVec, Mat, MutCol, MutDiag, MutRow, MutView, Row, RowVec, Scaled, ToBlasint, Trans,
    View,
};
use blas::axpy::Axpy;
use traits::{Matrix, MatrixCols, MatrixMutCols, MatrixMutRows, MatrixRows, SubAssign};

// vector -= scalar
fn vs<T>(lhs: ::raw::strided::Slice<T>, rhs: T) where T: Axpy + Neg<T> + One {
    let n = lhs.len;

    if n == 0 { return }

    let axpy = Axpy::axpy(None::<T>);
    let alpha: T = One::one();
    let alpha = -alpha;
    let x = &rhs;
    let incx = 0;
    let y = lhs.data;
    let incy = lhs.stride.to_blasint();

    unsafe { axpy(&n.to_blasint(), &alpha, x, &incx, y, &incy) }
}

// col
impl<T> SubAssign<T> for ColVec<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.as_mut_col().sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<T> for MutCol<'a, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: T) {
        vs((self.0).0, rhs)
    }
}

// diag
impl<'a, T> SubAssign<T> for MutDiag<'a, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: T) {
        vs((self.0).0, rhs)
    }
}

// mat
impl<T> SubAssign<T> for Mat<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.unroll_mut().sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<T> for MutView<'a, T> where T: Axpy + Clone + Neg<T> + One {
    fn sub_assign(&mut self, rhs: T) {
        if self.nrows() < self.ncols() {
            for mut row in self.mut_rows() {
                row.sub_assign(rhs.clone())
            }
        } else {
            for mut col in self.mut_cols() {
                col.sub_assign(rhs.clone())
            }
        }
    }
}

impl<T> SubAssign<T> for Trans<Mat<T>> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.0.sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<T> for Trans<MutView<'a, T>> where T: Axpy + Clone + Neg<T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.0.sub_assign(rhs)
    }
}

// row
impl<'a, T> SubAssign<T> for MutRow<'a, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: T) {
        vs((self.0).0, rhs)
    }
}

impl<T> SubAssign<T> for RowVec<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: T) {
        self.as_mut_row().sub_assign(rhs)
    }
}

// vector -= vector
fn vv<T>(lhs: ::raw::strided::Slice<T>, alpha: T, rhs: ::raw::strided::Slice<T>) where
    T: Axpy + Neg<T>,
{
    assert_eq!(lhs.len, rhs.len);

    let n = lhs.len;

    if n == 0 { return }

    let axpy = Axpy::axpy(None::<T>);
    let n = n.to_blasint();
    let alpha = -alpha;
    let x = rhs.data;
    let incx = rhs.stride.to_blasint();
    let y = lhs.data;
    let incy = lhs.stride.to_blasint();

    unsafe { axpy(&n, &alpha, x, &incx, y, &incy) }
}

// col
impl<'a, T> SubAssign<Col<'a, T>> for ColVec<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Col<T>) {
        self.as_mut_col().sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<&'a ColVec<T>> for ColVec<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &ColVec<T>) {
        self.as_mut_col().sub_assign(rhs.as_col())
    }
}

impl<'a, 'b, T> SubAssign<&'a MutCol<'b, T>> for ColVec<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &MutCol<T>) {
        self.as_mut_col().sub_assign(rhs.as_col())
    }
}

impl<'a, T> SubAssign<Scaled<T, Col<'a, T>>> for ColVec<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Scaled<T, Col<T>>) {
        self.as_mut_col().sub_assign(rhs)
    }
}

impl<'a, 'b, T> SubAssign<Col<'a, T>> for MutCol<'b, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Col<T>) {
        vv((self.0).0, One::one(), (rhs.0).0)
    }
}

impl<'a, 'b, T> SubAssign<&'a ColVec<T>> for MutCol<'b, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &ColVec<T>) {
        self.sub_assign(rhs.as_col())
    }
}

impl<'a, 'b, 'c, T> SubAssign<&'a MutCol<'b, T>> for MutCol<'c, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &MutCol<T>) {
        self.sub_assign(rhs.as_col())
    }
}

impl<'a, 'b, T> SubAssign<Scaled<T, Col<'a, T>>> for MutCol<'b, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Scaled<T, Col<T>>) {
        vv((self.0).0, rhs.0, ((rhs.1).0).0)
    }
}

// matrix -= matrix
macro_rules! mm {
    ($lhs:ty) => {
        fn sub_assign(&mut self, rhs: $lhs) {
            assert_eq!(self.size(), rhs.size());

            if self.nrows() < self.ncols() {
                for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                    lhs.sub_assign(rhs)
                }
            } else {
                for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                    lhs.sub_assign(rhs)
                }
            }
        }
    }
}

// mat
// XXX (nit) double "equal size" assertion
impl<'a, T> SubAssign<&'a Mat<T>> for Mat<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &Mat<T>) {
        assert_eq!(self.size(), rhs.size());

        self.unroll_mut().sub_assign(rhs.unroll())
    }
}

impl<'a, 'b, T> SubAssign<&'a MutView<'b, T>> for Mat<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &MutView<T>) {
        self.as_mut_view().sub_assign(rhs.as_view())
    }
}

impl<'a, T> SubAssign<Scaled<T, Trans<View<'a, T>>>> for Mat<T> where
    T: Axpy + Clone + Neg<T> + One,
{
    fn sub_assign(&mut self, rhs: Scaled<T, Trans<View<T>>>) {
        self.as_mut_view().sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<Scaled<T, View<'a, T>>> for Mat<T> where T: Axpy + Clone + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Scaled<T, View<T>>) {
        self.as_mut_view().sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<&'a Trans<Mat<T>>> for Mat<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.as_mut_view().sub_assign(Trans(rhs.0.as_view()))
    }
}

impl<'a, 'b, T> SubAssign<&'a Trans<MutView<'b, T>>> for Mat<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.as_mut_view().sub_assign(Trans(rhs.0.as_view()))
    }
}

impl<'a, 'b, T> SubAssign<Trans<View<'b, T>>> for Mat<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Trans<View<T>>) {
        self.as_mut_view().sub_assign(rhs)
    }
}

impl<'a, 'b, T> SubAssign<View<'a, T>> for Mat<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: View<T>) {
        self.as_mut_view().sub_assign(rhs)
    }
}

impl<'a, 'b, T> SubAssign<&'a Mat<T>> for MutView<'b, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &Mat<T>) {
        self.sub_assign(rhs.as_view())
    }
}

impl<'a, 'b, 'c, T> SubAssign<&'a MutView<'b, T>> for MutView<'c, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &MutView<T>) {
        self.sub_assign(rhs.as_view())
    }
}

impl<'a, 'b, T> SubAssign<Scaled<T, Trans<View<'a, T>>>> for MutView<'b, T> where
    T: Axpy + Clone + Neg<T> + One,
{
    mm!(Scaled<T, Trans<View<T>>>);
}

impl<'a, 'b, T> SubAssign<Scaled<T, View<'a, T>>> for MutView<'b, T> where
    T: Axpy + Clone + Neg<T> + One,
{
    mm!(Scaled<T, View<T>>);
}

impl<'a, 'b, T> SubAssign<&'a Trans<Mat<T>>> for MutView<'b, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.sub_assign(Trans(rhs.0.as_view()))
    }
}

impl<'a, 'b, 'c, T> SubAssign<&'a Trans<MutView<'b, T>>> for MutView<'c, T> where
    T: Axpy + Neg<T> + One,
{
    fn sub_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.sub_assign(Trans(rhs.0.as_view()))
    }
}

impl<'a, 'b, T> SubAssign<Trans<View<'a, T>>> for MutView<'b, T> where T: Axpy + Neg<T> + One {
    mm!(Trans<View<T>>);
}

impl<'a, 'b, T> SubAssign<View<'a, T>> for MutView<'b, T> where T: Axpy + Neg<T> + One {
    mm!(View<T>);
}

impl<'a, T> SubAssign<&'a Mat<T>> for Trans<Mat<T>> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &Mat<T>) {
        Trans(self.0.as_mut_view()).sub_assign(rhs.as_view())
    }
}

impl<'a, 'b, T> SubAssign<&'a MutView<'b, T>> for Trans<Mat<T>> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &MutView<T>) {
        Trans(self.0.as_mut_view()).sub_assign(rhs.as_view())
    }
}

impl<'a, T> SubAssign<Scaled<T, Trans<View<'a, T>>>> for Trans<Mat<T>> where
    T: Axpy + Clone + Neg<T> + One,
{
    fn sub_assign(&mut self, rhs: Scaled<T, Trans<View<T>>>) {
        self.0.as_mut_view().sub_assign(Scaled(rhs.0, (rhs.1).0))
    }
}

impl<'a, T> SubAssign<Scaled<T, View<'a, T>>> for Trans<Mat<T>> where
    T: Axpy + Clone + Neg<T> + One,
{
    fn sub_assign(&mut self, rhs: Scaled<T, View<T>>) {
        Trans(self.0.as_mut_view()).sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<&'a Trans<Mat<T>>> for Trans<Mat<T>> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.0.sub_assign(rhs.0.as_view())
    }
}

impl<'a, 'b, T> SubAssign<&'a Trans<MutView<'b, T>>> for Trans<Mat<T>> where
    T: Axpy + Neg<T> + One,
{
    fn sub_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.0.sub_assign(rhs.0.as_view())
    }
}

impl<'a, T> SubAssign<Trans<View<'a, T>>> for Trans<Mat<T>> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Trans<View<T>>) {
        self.0.sub_assign(rhs.0)
    }
}

impl<'a, T> SubAssign<View<'a, T>> for Trans<Mat<T>> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: View<T>) {
        Trans(self.0.as_mut_view()).sub_assign(rhs)
    }
}

impl<'a, 'b, T> SubAssign<&'a Mat<T>> for Trans<MutView<'b, T>> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &Mat<T>) {
        self.sub_assign(rhs.as_view())
    }
}

impl<'a, 'b, 'c, T> SubAssign<&'a MutView<'b, T>> for Trans<MutView<'c, T>> where
    T: Axpy + Neg<T> + One,
{
    fn sub_assign(&mut self, rhs: &MutView<T>) {
        self.sub_assign(rhs.as_view())
    }
}

impl<'a, 'b, T> SubAssign<Scaled<T, Trans<View<'a, T>>>> for Trans<MutView<'b, T>> where
    T: Axpy + Clone + Neg<T> + One,
{
    fn sub_assign(&mut self, rhs: Scaled<T, Trans<View<T>>>) {
        self.0.sub_assign(Scaled(rhs.0, (rhs.1).0))
    }
}

impl<'a, 'b, T> SubAssign<Scaled<T, View<'a, T>>> for Trans<MutView<'b, T>> where
    T: Axpy + Clone + Neg<T> + One,
{
    mm!(Scaled<T, View<T>>);
}

impl<'a, 'b, T> SubAssign<&'a Trans<Mat<T>>> for Trans<MutView<'b, T>> where
    T: Axpy + Neg<T> + One,
{
    fn sub_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.0.sub_assign(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> SubAssign<&'a Trans<MutView<'b, T>>> for Trans<MutView<'c, T>> where
    T: Axpy + Neg<T> + One,
{
    fn sub_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.0.sub_assign(rhs.0.as_view())
    }
}

impl<'a, 'b, T> SubAssign<Trans<View<'a, T>>> for Trans<MutView<'b, T>> where
    T: Axpy + Neg<T> + One,
{
    fn sub_assign(&mut self, rhs: Trans<View<T>>) {
        self.0.sub_assign(rhs.0)
    }
}

impl<'a, 'b, T> SubAssign<View<'a, T>> for Trans<MutView<'b, T>> where T: Axpy + Neg<T> + One {
    mm!(View<T>);
}

// row
impl<'a, 'b, 'c, T> SubAssign<&'a MutRow<'b, T>> for MutRow<'c, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &MutRow<T>) {
        self.sub_assign(rhs.as_row())
    }
}

impl<'a, 'b, T> SubAssign<Row<'a, T>> for MutRow<'b, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Row<T>) {
        vv((self.0).0, One::one(), (rhs.0).0)
    }
}

impl<'a, 'b, T> SubAssign<&'a RowVec<T>> for MutRow<'b, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &RowVec<T>) {
        self.sub_assign(rhs.as_row())
    }
}

impl<'a, 'b, T> SubAssign<Scaled<T, Row<'a, T>>> for MutRow<'b, T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Scaled<T, Row<T>>) {
        vv((self.0).0, rhs.0, ((rhs.1).0).0)
    }
}

impl<'a, 'b, T> SubAssign<&'a MutRow<'b, T>> for RowVec<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &MutRow<T>) {
        self.as_mut_row().sub_assign(rhs.as_row())
    }
}

impl<'a, T> SubAssign<Row<'a, T>> for RowVec<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Row<T>) {
        self.as_mut_row().sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<&'a RowVec<T>> for RowVec<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: &RowVec<T>) {
        self.as_mut_row().sub_assign(rhs.as_row())
    }
}

impl<'a, T> SubAssign<Scaled<T, Row<'a, T>>> for RowVec<T> where T: Axpy + Neg<T> + One {
    fn sub_assign(&mut self, rhs: Scaled<T, Row<T>>) {
        self.as_mut_row().sub_assign(rhs)
    }
}
