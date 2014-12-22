use std::iter::order;
use traits::{Iter, Matrix, MatrixRows};
use {Col, ColVec, Diag, Mat, MutCol, MutDiag, MutRow, MutView, Row, RowVec, Trans, View};

// col
impl<'a, 'b, T, U> PartialEq<Col<'a, T>> for Col<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Col<T>) -> bool {
        (self.0).0 == (rhs.0).0
    }
}

impl<'a, T, U> PartialEq<ColVec<T>> for Col<'a, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &ColVec<T>) -> bool {
        *self == rhs.as_col()
    }
}

impl<'a, 'b, T, U> PartialEq<MutCol<'a, T>> for Col<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutCol<T>) -> bool {
        *self == rhs.as_col()
    }
}

impl<'a, T, U> PartialEq<Col<'a, T>> for ColVec<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Col<T>) -> bool {
        self.as_col() == *rhs
    }
}

impl<T, U> PartialEq<ColVec<T>> for ColVec<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &ColVec<T>) -> bool {
        self.as_col() == rhs.as_col()
    }
}

impl<'a, T, U> PartialEq<MutCol<'a, T>> for ColVec<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutCol<T>) -> bool {
        self.as_col() == rhs.as_col()
    }
}

impl<'a, 'b, T, U> PartialEq<Col<'a, T>> for MutCol<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Col<T>) -> bool {
        self.as_col() == *rhs
    }
}

impl<'a, T, U> PartialEq<ColVec<T>> for MutCol<'a, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &ColVec<T>) -> bool {
        self.as_col() == rhs.as_col()
    }
}

impl<'a, 'b, T, U> PartialEq<MutCol<'a, T>> for MutCol<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutCol<T>) -> bool {
        self.as_col() == rhs.as_col()
    }
}

// diag
impl<'a, 'b, T, U> PartialEq<Diag<'a, T>> for Diag<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Diag<T>) -> bool {
        (self.0).0 == (rhs.0).0
    }
}

impl<'a, 'b, T, U> PartialEq<MutDiag<'a, T>> for Diag<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutDiag<T>) -> bool {
        *self == rhs.as_diag()
    }
}

impl<'a, 'b, T, U> PartialEq<Diag<'a, T>> for MutDiag<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Diag<T>) -> bool {
        self.as_diag() == *rhs
    }
}

impl<'a, 'b, T, U> PartialEq<MutDiag<'a, T>> for MutDiag<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutDiag<T>) -> bool {
        self.as_diag() == rhs.as_diag()
    }
}

// mat
impl<T, U> PartialEq<Mat<T>> for Mat<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Mat<T>) -> bool {
        self.size() == rhs.size() && order::eq(self.data.iter(), rhs.data.iter())
    }
}

impl<'a, T, U> PartialEq<MutView<'a, T>> for Mat<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutView<T>) -> bool {
        self.as_view() == rhs.as_view()
    }
}

impl<T, U> PartialEq<Trans<Mat<T>>> for Mat<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<Mat<T>>) -> bool {
        self.as_view() == Trans(rhs.0.as_view())
    }
}

impl<'a, T, U> PartialEq<Trans<MutView<'a, T>>> for Mat<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<MutView<T>>) -> bool {
        self.as_view() == Trans(rhs.0.as_view())
    }
}

impl<'a, T, U> PartialEq<Trans<View<'a, T>>> for Mat<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<View<T>>) -> bool {
        self.as_view() == *rhs
    }
}

impl<'a, T, U> PartialEq<View<'a, T>> for Mat<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &View<T>) -> bool {
        self.as_view() == *rhs
    }
}

impl<'a, T, U> PartialEq<Mat<T>> for MutView<'a, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Mat<T>) -> bool {
        self.as_view() == rhs.as_view()
    }
}

impl<'a, 'b, T, U> PartialEq<MutView<'a, T>> for MutView<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutView<T>) -> bool {
        self.as_view() == rhs.as_view()
    }
}

impl<'a, T, U> PartialEq<Trans<Mat<T>>> for MutView<'a, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<Mat<T>>) -> bool {
        self.as_view() == Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, T, U> PartialEq<Trans<MutView<'a, T>>> for MutView<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<MutView<T>>) -> bool {
        self.as_view() == Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, T, U> PartialEq<Trans<View<'a, T>>> for MutView<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<View<T>>) -> bool {
        self.as_view() == *rhs
    }
}

impl<'a, 'b, T, U> PartialEq<View<'a, T>> for MutView<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &View<T>) -> bool {
        self.as_view() == *rhs
    }
}

impl<T, U> PartialEq<Mat<T>> for Trans<Mat<U>> where T: PartialEq<U> {
    fn eq(&self, rhs: &Mat<T>) -> bool {
        rhs.as_view() == Trans(self.0.as_view())
    }
}

impl<'a, T, U> PartialEq<MutView<'a, T>> for Trans<Mat<U>> where T: PartialEq<U> {
    fn eq(&self, rhs: &MutView<T>) -> bool {
        rhs.as_view() == Trans(self.0.as_view())
    }
}

impl<T, U> PartialEq<Trans<Mat<T>>> for Trans<Mat<U>> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<Mat<T>>) -> bool {
        self.0.as_view() == rhs.0.as_view()
    }
}

impl<'a, T, U> PartialEq<Trans<MutView<'a, T>>> for Trans<Mat<U>> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<MutView<T>>) -> bool {
        self.0.as_view() == rhs.0.as_view()
    }
}

impl<'a, T, U> PartialEq<Trans<View<'a, T>>> for Trans<Mat<U>> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<View<T>>) -> bool {
        self.0.as_view() == rhs.0
    }
}

impl<'a, T, U> PartialEq<View<'a, T>> for Trans<Mat<U>> where T: PartialEq<U> {
    fn eq(&self, rhs: &View<T>) -> bool {
        *rhs == Trans(self.0.as_view())
    }
}

impl<'a, T, U> PartialEq<Mat<T>> for Trans<MutView<'a, U>> where T: PartialEq<U> {
    fn eq(&self, rhs: &Mat<T>) -> bool {
        rhs.as_view() == Trans(self.0.as_view())
    }
}

impl<'a, 'b, T, U> PartialEq<MutView<'a, T>> for Trans<MutView<'b, U>> where T: PartialEq<U> {
    fn eq(&self, rhs: &MutView<T>) -> bool {
        rhs.as_view() == Trans(self.0.as_view())
    }
}

impl<'a, T, U> PartialEq<Trans<Mat<T>>> for Trans<MutView<'a, U>> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<Mat<T>>) -> bool {
        self.0.as_view() == rhs.0.as_view()
    }
}

impl<'a, 'b, T, U> PartialEq<Trans<MutView<'a, T>>> for Trans<MutView<'b, U>> where
    U: PartialEq<T>,
{
    fn eq(&self, rhs: &Trans<MutView<T>>) -> bool {
        self.0.as_view() == rhs.0.as_view()
    }
}

impl<'a, 'b, T, U> PartialEq<Trans<View<'a, T>>> for Trans<MutView<'b, U>> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<View<T>>) -> bool {
        self.0.as_view() == rhs.0
    }
}

impl<'a, 'b, T, U> PartialEq<View<'a, T>> for Trans<MutView<'b, U>> where T: PartialEq<U> {
    fn eq(&self, rhs: &View<T>) -> bool {
        *rhs == Trans(self.0.as_view())
    }
}

impl<'a, T, U> PartialEq<Mat<T>> for Trans<View<'a, U>> where T: PartialEq<U> {
    fn eq(&self, rhs: &Mat<T>) -> bool {
        rhs.as_view() == *self
    }
}

impl<'a, 'b, T, U> PartialEq<MutView<'a, T>> for Trans<View<'b, U>> where T: PartialEq<U> {
    fn eq(&self, rhs: &MutView<T>) -> bool {
        rhs.as_view() == *self
    }
}

impl<'a, T, U> PartialEq<Trans<Mat<T>>> for Trans<View<'a, U>> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<Mat<T>>) -> bool {
        self.0 == rhs.0.as_view()
    }
}

impl<'a, 'b, T, U> PartialEq<Trans<MutView<'a, T>>> for Trans<View<'b, U>> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<MutView<T>>) -> bool {
        self.0 == rhs.0.as_view()
    }
}

impl<'a, 'b, T, U> PartialEq<Trans<View<'a, T>>> for Trans<View<'b, U>> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<View<T>>) -> bool {
        self.0 == rhs.0
    }
}

impl<'a, 'b, T, U> PartialEq<View<'a, T>> for Trans<View<'b, U>> where T: PartialEq<U> {
    fn eq(&self, rhs: &View<T>) -> bool {
        *rhs == *self
    }
}

impl<'a, T, U> PartialEq<Mat<T>> for View<'a, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Mat<T>) -> bool {
        *self == rhs.as_view()
    }
}

impl<'a, 'b, T, U> PartialEq<MutView<'a, T>> for View<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutView<T>) -> bool {
        *self == rhs.as_view()
    }
}

impl<'a, T, U> PartialEq<Trans<Mat<T>>> for View<'a, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<Mat<T>>) -> bool {
        *self == Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, T, U> PartialEq<Trans<MutView<'a, T>>> for View<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<MutView<T>>) -> bool {
        *self == Trans(rhs.0.as_view())
    }
}

// XXX Is this the fastest way to check for equality?
impl<'a, 'b, T, U> PartialEq<Trans<View<'a, T>>> for View<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Trans<View<T>>) -> bool {
        self.size() == rhs.size() && self.rows().zip(rhs.rows()).all(|(lhs_row, rhs_row)| {
            order::eq(lhs_row.iter(), rhs_row.iter())
        })
    }
}

impl<'a, 'b, T, U> PartialEq<View<'a, T>> for View<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &View<T>) -> bool {
        self.0 == rhs.0
    }
}

// row
impl<'a, 'b, T, U> PartialEq<Row<'a, T>> for Row<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Row<T>) -> bool {
        (self.0).0 == (rhs.0).0
    }
}

impl<'a, T, U> PartialEq<RowVec<T>> for Row<'a, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &RowVec<T>) -> bool {
        *self == rhs.as_row()
    }
}

impl<'a, 'b, T, U> PartialEq<MutRow<'a, T>> for Row<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutRow<T>) -> bool {
        *self == rhs.as_row()
    }
}

impl<'a, T, U> PartialEq<Row<'a, T>> for RowVec<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Row<T>) -> bool {
        self.as_row() == *rhs
    }
}

impl<T, U> PartialEq<RowVec<T>> for RowVec<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &RowVec<T>) -> bool {
        self.as_row() == rhs.as_row()
    }
}

impl<'a, T, U> PartialEq<MutRow<'a, T>> for RowVec<U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutRow<T>) -> bool {
        self.as_row() == rhs.as_row()
    }
}

impl<'a, 'b, T, U> PartialEq<Row<'a, T>> for MutRow<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &Row<T>) -> bool {
        self.as_row() == *rhs
    }
}

impl<'a, T, U> PartialEq<RowVec<T>> for MutRow<'a, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &RowVec<T>) -> bool {
        self.as_row() == rhs.as_row()
    }
}

impl<'a, 'b, T, U> PartialEq<MutRow<'a, T>> for MutRow<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &MutRow<T>) -> bool {
        self.as_row() == rhs.as_row()
    }
}
