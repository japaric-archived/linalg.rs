use std::slice;

use {Col, Diag, Mat, Row};
use traits::{Iter, IterMut};

impl<'a, T> Iter<'a, &'a T, slice::Items<'a, T>> for Mat<T> {
    fn iter(&self) -> slice::Items<T> {
        self.data.iter()
    }
}

impl<'a, T> IterMut<'a, &'a mut T, slice::MutItems<'a, T>> for Mat<T> {
    fn iter_mut(&mut self) -> slice::MutItems<T> {
        self.data.iter_mut()
    }
}

impl<'a, T> Iter<'a, &'a T, slice::Items<'a, T>> for Box<[T]> {
    fn iter(&self) -> slice::Items<T> {
        SliceExt::iter(&**self)
    }
}

impl<'a, T> IterMut<'a, &'a mut T, slice::MutItems<'a, T>> for Box<[T]> {
    fn iter_mut(&mut self) -> slice::MutItems<T> {
        SliceExt::iter_mut(&mut **self)
    }
}

impl<'a, 'b, T> IterMut<'b, &'b mut T, slice::MutItems<'b, T>> for &'a mut [T] {
    fn iter_mut(&mut self) -> slice::MutItems<T> {
        SliceExt::iter_mut(*self)
    }
}

macro_rules! impl_iter {
    ($($ty:ty),+) => {$(
        impl<'a, 'b, T> Iter<'b, &'b T, slice::Items<'b, T>> for $ty {
            fn iter(&self) -> slice::Items<T> {
                SliceExt::iter(*self)
            }
        })+
    }
}

impl_iter!(&'a [T], &'a mut [T]);

macro_rules! impls {
    ($($ty:ty),+) => {$(
        impl<'a, T, I, V> Iter<'a, T, I> for $ty where
            I: Iterator<T>,
            V: Iter<'a, T, I>,
        {
            fn iter(&'a self) -> I {
                self.0.iter()
            }
        }

        impl<'a, T, I, V> IterMut<'a, T, I> for $ty where
            I: Iterator<T>,
            V: IterMut<'a, T, I>,
        {
            fn iter_mut(&'a mut self) -> I {
                self.0.iter_mut()
            }
        })+
   }
}

impls!(Col<V>, Diag<V>, Row<V>);
