use blas::copy::BlasCopy;
use blas::{BlasPtr, BlasStride, to_blasint};
use std::slice;

pub trait PrivateIter<'a, T, I> where I: Iterator<T> {
    fn private_iter(&'a self) -> I;
}

macro_rules! private_iter {
    () => {
        fn private_iter(&self) -> slice::Items<T> {
            self.iter()
        }
    }
}

impl<'a, 'b, T> PrivateIter<'b, &'b T, slice::Items<'b, T>> for &'a [T] { private_iter!() }
impl<'a, 'b, T> PrivateIter<'b, &'b T, slice::Items<'b, T>> for &'a mut [T] { private_iter!() }
impl<'a, T> PrivateIter<'a, &'a T, slice::Items<'a, T>> for Vec<T> { private_iter!() }

pub trait PrivateMutIter<'a, T, I> where I: Iterator<T> {
    fn private_mut_iter(&'a mut self) -> I;
}

macro_rules! private_mut_iter {
    () => {
        fn private_mut_iter(&mut self) -> slice::MutItems<T> {
            self.mut_iter()
        }
    }
}

impl<'a, 'b, T> PrivateMutIter<'b, &'b mut T, slice::MutItems<'b, T>> for &'a mut [T] {
    private_mut_iter!()
}

impl<'a, T> PrivateMutIter<'a, &'a mut T, slice::MutItems<'a, T>> for Vec<T> {
    private_mut_iter!()
}

pub trait PrivateToOwned<T>: BlasPtr<T> + BlasStride + Collection where T: BlasCopy {
    fn private_to_owned(&self) -> Vec<T> {
        let length = self.len();
        let mut data = Vec::with_capacity(length);
        unsafe { data.set_len(length) }

        let n = &to_blasint(length);
        let x = self.blas_ptr();
        let incx = &self.blas_stride();

        let ffi = BlasCopy::copy(None::<T>);
        unsafe { ffi(n, x, incx, data.as_mut_ptr(), &1) }

        data
    }
}

impl<'a, T> PrivateToOwned<T> for &'a [T] where T: BlasCopy {}
impl<'a, T> PrivateToOwned<T> for &'a mut [T] where T: BlasCopy {}
