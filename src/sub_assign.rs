use std::ops::Neg;

use assign::SubAssign;
use blas::{Axpy, blasint};
use cast::CastTo;
use onezero::One;

use {
    Col, ColVec, Mat, MutCol, MutDiag, MutRow, MutView, Row, RowVec, Scaled, Trans,
    View,
};
use traits::{Matrix, MatrixCols, MatrixMutCols, MatrixMutRows, MatrixRows};

// vector -= scalar
fn vs<T>(lhs: ::raw::strided::Slice<T>, rhs: &T) where T: Axpy + Neg<Output=T> + One {
    let n = lhs.len;

    if n == 0 { return }

    let axpy = T::axpy();
    let alpha: T = One::one();
    let alpha = -alpha;
    let x = rhs;
    let incx = 0;
    let y = lhs.data;
    let incy = lhs.stride.to::<blasint>().unwrap();

    unsafe { axpy(&n.to::<blasint>().unwrap(), &alpha, x, &incx, y, &incy) }
}

// col
impl<T> SubAssign<T> for ColVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        self.as_mut_col().sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<T> for MutCol<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        vs((self.0).0, rhs)
    }
}

// diag
impl<'a, T> SubAssign<T> for MutDiag<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        vs((self.0).0, rhs)
    }
}

// mat
impl<T> SubAssign<T> for Mat<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        self.unroll_mut().sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<T> for MutView<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        if self.nrows() < self.ncols() {
            for mut row in self.mut_rows() {
                row.sub_assign(rhs)
            }
        } else {
            for mut col in self.mut_cols() {
                col.sub_assign(rhs)
            }
        }
    }
}

impl<T> SubAssign<T> for Trans<Mat<T>> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        self.0.sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<T> for Trans<MutView<'a, T>> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        self.0.sub_assign(rhs)
    }
}

// row
impl<'a, T> SubAssign<T> for MutRow<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        vs((self.0).0, rhs)
    }
}

impl<T> SubAssign<T> for RowVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &T) {
        self.as_mut_row().sub_assign(rhs)
    }
}

// vector -= vector
fn vv<T>(lhs: ::raw::strided::Slice<T>, alpha: T, rhs: ::raw::strided::Slice<T>) where
    T: Axpy + Neg<Output=T>,
{
    assert_eq!(lhs.len, rhs.len);

    let n = lhs.len;

    if n == 0 { return }

    let axpy = T::axpy();
    let n = n.to::<blasint>().unwrap();
    let alpha = -alpha;
    let x = rhs.data;
    let incx = rhs.stride.to::<blasint>().unwrap();
    let y = lhs.data;
    let incy = lhs.stride.to::<blasint>().unwrap();

    unsafe { axpy(&n, &alpha, x, &incx, y, &incy) }
}

// col
impl<'a, T> SubAssign<Col<'a, T>> for ColVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Col<T>) {
        self.as_mut_col().sub_assign(rhs)
    }
}

impl<T> SubAssign<ColVec<T>> for ColVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &ColVec<T>) {
        self.as_mut_col().sub_assign(&rhs.as_col())
    }
}

impl<'a, T> SubAssign<MutCol<'a, T>> for ColVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &MutCol<T>) {
        self.as_mut_col().sub_assign(rhs.as_col())
    }
}

impl<'a, T> SubAssign<Scaled<T, Col<'a, T>>> for ColVec<T> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Scaled<T, Col<'a, T>>) {
        self.as_mut_col().sub_assign(rhs)
    }
}

impl<'a, 'b, T> SubAssign<Col<'a, T>> for MutCol<'b, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Col<T>) {
        vv((self.0).0, One::one(), (rhs.0).0)
    }
}

impl<'a, T> SubAssign<ColVec<T>> for MutCol<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &ColVec<T>) {
        self.sub_assign(&rhs.as_col())
    }
}

impl<'a, 'b, T> SubAssign<MutCol<'a, T>> for MutCol<'b, T> where
    T: Axpy + Neg<Output=T> + One
{
    fn sub_assign(&mut self, rhs: &MutCol<T>) {
        self.sub_assign(rhs.as_col())
    }
}

impl<'a, 'b, T> SubAssign<Scaled<T, Col<'a, T>>> for MutCol<'b, T> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One
{
    fn sub_assign(&mut self, rhs: &Scaled<T, Col<'a, T>>) {
        vv((self.0).0, rhs.0.clone(), ((rhs.1).0).0)
    }
}

// matrix -= matrix
macro_rules! mm {
    ($lhs:ty) => {
        fn sub_assign(&mut self, rhs: &$lhs) {
            assert_eq!(self.size(), rhs.size());

            if self.nrows() < self.ncols() {
                for (mut lhs, rhs) in self.mut_rows().zip(rhs.rows()) {
                    lhs.sub_assign(&rhs)
                }
            } else {
                for (mut lhs, rhs) in self.mut_cols().zip(rhs.cols()) {
                    lhs.sub_assign(&rhs)
                }
            }
        }
    }
}

// mat
// XXX (nit) two "equal size" assertions
impl<T> SubAssign<Mat<T>> for Mat<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Mat<T>) {
        assert_eq!(self.size(), rhs.size());

        self.unroll_mut().sub_assign(&rhs.unroll())
    }
}

impl<'a, T> SubAssign<MutView<'a, T>> for Mat<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &MutView<T>) {
        self.as_mut_view().sub_assign(rhs.as_view())
    }
}

impl<'a, T> SubAssign<Scaled<T, Trans<View<'a, T>>>> for Mat<T> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Scaled<T, Trans<View<'a, T>>>) {
        self.as_mut_view().sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<Scaled<T, View<'a, T>>> for Mat<T> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One
{
    fn sub_assign(&mut self, rhs: &Scaled<T, View<'a, T>>) {
        self.as_mut_view().sub_assign(rhs)
    }
}

impl<T> SubAssign<Trans<Mat<T>>> for Mat<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.as_mut_view().sub_assign(&Trans(rhs.0.as_view()))
    }
}

impl<'a, T> SubAssign<Trans<MutView<'a, T>>> for Mat<T> where
    T: Axpy + Neg<Output=T> + One
{
    fn sub_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.as_mut_view().sub_assign(rhs.as_trans_view())
    }
}

impl<'a, 'b, T> SubAssign<Trans<View<'b, T>>> for Mat<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Trans<View<T>>) {
        self.as_mut_view().sub_assign(rhs)
    }
}

impl<'a, 'b, T> SubAssign<View<'a, T>> for Mat<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &View<T>) {
        self.as_mut_view().sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<Mat<T>> for MutView<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Mat<T>) {
        self.sub_assign(&rhs.as_view())
    }
}

impl<'a, 'b, T> SubAssign<MutView<'a, T>> for MutView<'b, T> where
    T: Axpy + Neg<Output=T> + One
{
    fn sub_assign(&mut self, rhs: &MutView<T>) {
        self.sub_assign(rhs.as_view())
    }
}

impl<'a, 'b, T> SubAssign<Scaled<T, Trans<View<'a, T>>>> for MutView<'b, T> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One,
{
    mm!(Scaled<T, Trans<View<'a, T>>>);
}

impl<'a, 'b, T> SubAssign<Scaled<T, View<'a, T>>> for MutView<'b, T> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One,
{
    mm!(Scaled<T, View<'a, T>>);
}

impl<'a, T> SubAssign<Trans<Mat<T>>> for MutView<'a, T> where
    T: Axpy + Neg<Output=T> + One
{
    fn sub_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.sub_assign(&Trans(rhs.0.as_view()))
    }
}

impl<'a, 'b, T> SubAssign<Trans<MutView<'a, T>>> for MutView<'b, T> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.sub_assign(rhs.as_trans_view())
    }
}

impl<'a, 'b, T> SubAssign<Trans<View<'a, T>>> for MutView<'b, T> where
    T: Axpy + Neg<Output=T> + One
{
    mm!(Trans<View<T>>);
}

impl<'a, 'b, T> SubAssign<View<'a, T>> for MutView<'b, T> where T: Axpy + Neg<Output=T> + One {
    mm!(View<T>);
}

impl<T> SubAssign<Mat<T>> for Trans<Mat<T>> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Mat<T>) {
        Trans(self.0.as_mut_view()).sub_assign(&rhs.as_view())
    }
}

impl<'a, T> SubAssign<MutView<'a, T>> for Trans<Mat<T>> where
    T: Axpy + Neg<Output=T> + One
{
    fn sub_assign(&mut self, rhs: &MutView<T>) {
        Trans(self.0.as_mut_view()).sub_assign(rhs.as_view())
    }
}

impl<'a, T> SubAssign<Scaled<T, Trans<View<'a, T>>>> for Trans<Mat<T>> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Scaled<T, Trans<View<'a, T>>>) {
        self.0.as_mut_view().sub_assign(&Scaled(rhs.0.clone(), (rhs.1).0))
    }
}

impl<'a, T> SubAssign<Scaled<T, View<'a, T>>> for Trans<Mat<T>> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Scaled<T, View<'a, T>>) {
        Trans(self.0.as_mut_view()).sub_assign(rhs)
    }
}

impl<T> SubAssign<Trans<Mat<T>>> for Trans<Mat<T>> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.0.sub_assign(&rhs.0.as_view())
    }
}

impl<'a, T> SubAssign<Trans<MutView<'a, T>>> for Trans<Mat<T>> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.0.sub_assign(rhs.0.as_view())
    }
}

impl<'a, T> SubAssign<Trans<View<'a, T>>> for Trans<Mat<T>> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Trans<View<T>>) {
        self.0.sub_assign(&rhs.0)
    }
}

impl<'a, T> SubAssign<View<'a, T>> for Trans<Mat<T>> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &View<T>) {
        Trans(self.0.as_mut_view()).sub_assign(rhs)
    }
}

impl<'a, T> SubAssign<Mat<T>> for Trans<MutView<'a, T>>
    where T: Axpy + Neg<Output=T> + One
{
    fn sub_assign(&mut self, rhs: &Mat<T>) {
        self.sub_assign(&rhs.as_view())
    }
}

impl<'a, 'b, T> SubAssign<MutView<'a, T>> for Trans<MutView<'b, T>> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &MutView<T>) {
        self.sub_assign(rhs.as_view())
    }
}

impl<'a, 'b, T> SubAssign<Scaled<T, Trans<View<'a, T>>>> for Trans<MutView<'b, T>> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Scaled<T, Trans<View<'a, T>>>) {
        self.0.sub_assign(&Scaled(rhs.0.clone(), (rhs.1).0))
    }
}

impl<'a, 'b, T> SubAssign<Scaled<T, View<'a, T>>> for Trans<MutView<'b, T>> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One,
{
    mm!(Scaled<T, View<'a, T>>);
}

impl<'a, T> SubAssign<Trans<Mat<T>>> for Trans<MutView<'a, T>> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Trans<Mat<T>>) {
        self.0.sub_assign(&rhs.0.as_view())
    }
}

impl<'a, 'b, T> SubAssign<Trans<MutView<'a, T>>> for Trans<MutView<'b, T>> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Trans<MutView<T>>) {
        self.0.sub_assign(rhs.0.as_view())
    }
}

impl<'a, 'b, T> SubAssign<Trans<View<'a, T>>> for Trans<MutView<'b, T>> where
    T: Axpy + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Trans<View<T>>) {
        self.0.sub_assign(&rhs.0)
    }
}

impl<'a, 'b, T> SubAssign<View<'a, T>> for Trans<MutView<'b, T>> where
    T: Axpy + Neg<Output=T> + One
{
    mm!(View<T>);
}

// row
impl<'a, 'b, T> SubAssign<MutRow<'a, T>> for MutRow<'b, T> where
    T: Axpy + Neg<Output=T> + One
{
    fn sub_assign(&mut self, rhs: &MutRow<T>) {
        self.sub_assign(rhs.as_row())
    }
}

impl<'a, 'b, T> SubAssign<Row<'a, T>> for MutRow<'b, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Row<T>) {
        vv((self.0).0, One::one(), (rhs.0).0)
    }
}

impl<'a, T> SubAssign<RowVec<T>> for MutRow<'a, T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &RowVec<T>) {
        self.sub_assign(&rhs.as_row())
    }
}

impl<'a, 'b, T> SubAssign<Scaled<T, Row<'a, T>>> for MutRow<'b, T> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One
{
    fn sub_assign(&mut self, rhs: &Scaled<T, Row<'a, T>>) {
        vv((self.0).0, rhs.0.clone(), ((rhs.1).0).0)
    }
}

impl<'a, T> SubAssign<MutRow<'a, T>> for RowVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &MutRow<T>) {
        self.as_mut_row().sub_assign(rhs.as_row())
    }
}

impl<'a, T> SubAssign<Row<'a, T>> for RowVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &Row<T>) {
        self.as_mut_row().sub_assign(rhs)
    }
}

impl<T> SubAssign<RowVec<T>> for RowVec<T> where T: Axpy + Neg<Output=T> + One {
    fn sub_assign(&mut self, rhs: &RowVec<T>) {
        self.as_mut_row().sub_assign(&rhs.as_row())
    }
}

impl<'a, T> SubAssign<Scaled<T, Row<'a, T>>> for RowVec<T> where
    T: 'a + Axpy + Clone + Neg<Output=T> + One,
{
    fn sub_assign(&mut self, rhs: &Scaled<T, Row<'a, T>>) {
        self.as_mut_row().sub_assign(rhs)
    }
}
