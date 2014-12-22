use std::slice;

use strided;
use traits::{Iter, IterMut};
use {Col, ColVec, Diag, Mat, MutCol, MutDiag, MutRow, Row, RowVec};

impl<'a, T> Iter<'a, &'a T, slice::Items<'a, T>> for ColVec<T> {
    fn iter(&'a self) -> slice::Items<'a, T> {
        self.0.iter()
    }
}

impl<'a, T> Iter<'a, &'a T, slice::Items<'a, T>> for Mat<T> {
    fn iter(&'a self) -> slice::Items<'a, T> {
        self.data.iter()
    }
}

impl<'a, T> Iter<'a, &'a T, slice::Items<'a, T>> for RowVec<T> {
    fn iter(&'a self) -> slice::Items<'a, T> {
        self.0.iter()
    }
}

macro_rules! impls {
    ($($ty:ty),+) => {
        $(
            impl<'a, 'b, T> Iter<'a, &'a T, strided::Items<'a, T>> for $ty {
                fn iter(&'a self) -> strided::Items<'a, T> {
                    self.0.iter()
                }
            }
        )+
    }
}

impls!(Col<'b, T>, Diag<'b, T>, MutCol<'b, T>, MutDiag<'b, T>, MutRow<'b, T>, Row<'b, T>);

impl<'a, T> IterMut<'a, &'a mut T, slice::MutItems<'a, T>> for ColVec<T> {
    fn iter_mut(&'a mut self) -> slice::MutItems<'a, T> {
        self.0.iter_mut()
    }
}

impl<'a, T> IterMut<'a, &'a mut T, slice::MutItems<'a, T>> for Mat<T> {
    fn iter_mut(&'a mut self) -> slice::MutItems<'a, T> {
        self.data.iter_mut()
    }
}

impl<'a, T> IterMut<'a, &'a mut T, slice::MutItems<'a, T>> for RowVec<T> {
    fn iter_mut(&'a mut self) -> slice::MutItems<'a, T> {
        self.0.iter_mut()
    }
}

macro_rules! impls_mut {
    ($($ty:ty),+) => {
        $(
            impl<'a, 'b, T> IterMut<'a, &'a mut T, strided::MutItems<'a, T>> for $ty {
                fn iter_mut(&'a mut self) -> strided::MutItems<'a, T> {
                    self.0.iter_mut()
                }
            }
        )+
    }
}

impls_mut!(MutCol<'b, T>, MutDiag<'b, T>, MutRow<'b, T>);
