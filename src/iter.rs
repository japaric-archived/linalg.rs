use std::slice;

use strided;
use traits::{Iter, IterMut};
use {Col, ColVec, Diag, Mat, MutCol, MutDiag, MutRow, Row, RowVec};

impl<'a, T> Iter<'a> for ColVec<T> {
    type Iter = slice::Iter<'a, T>;

    fn iter(&'a self) -> slice::Iter<'a, T> {
        self.0.iter()
    }
}

impl<'a, T> Iter<'a> for Mat<T> {
    type Iter = slice::Iter<'a, T>;

    fn iter(&'a self) -> slice::Iter<'a, T> {
        self.data.iter()
    }
}

impl<'a, T> Iter<'a> for RowVec<T> {
    type Iter = slice::Iter<'a, T>;

    fn iter(&'a self) -> slice::Iter<'a, T> {
        self.0.iter()
    }
}

macro_rules! impls {
    ($($ty:ty),+) => {
        $(
            impl<'a, 'b, T> Iter<'a> for $ty {
                type Iter = strided::Items<'a, T>;

                fn iter(&'a self) -> strided::Items<'a, T> {
                    self.0.iter()
                }
            }
        )+
    }
}

impls!(Col<'b, T>, Diag<'b, T>, MutCol<'b, T>, MutDiag<'b, T>, MutRow<'b, T>, Row<'b, T>);

impl<'a, T> IterMut<'a> for ColVec<T> {
    type Iter = slice::IterMut<'a, T>;

    fn iter_mut(&'a mut self) -> slice::IterMut<'a, T> {
        self.0.iter_mut()
    }
}

impl<'a, T> IterMut<'a> for Mat<T> {
    type Iter = slice::IterMut<'a, T>;

    fn iter_mut(&'a mut self) -> slice::IterMut<'a, T> {
        self.data.iter_mut()
    }
}

impl<'a, T> IterMut<'a> for RowVec<T> {
    type Iter = slice::IterMut<'a, T>;

    fn iter_mut(&'a mut self) -> slice::IterMut<'a, T> {
        self.0.iter_mut()
    }
}

macro_rules! impls_mut {
    ($($ty:ty),+) => {
        $(
            impl<'a, 'b, T> IterMut<'a> for $ty {
                type Iter = strided::MutItems<'a, T>;

                fn iter_mut(&'a mut self) -> strided::MutItems<'a, T> {
                    self.0.iter_mut()
                }
            }
        )+
    }
}

impls_mut!(MutCol<'b, T>, MutDiag<'b, T>, MutRow<'b, T>);
