use std::num::{One, mod};

use blas::axpy::BlasAxpy;
use blas::copy::BlasCopy;
use blas::{BlasPtr, BlasStride, to_blasint};
use notsafe::UnsafeMatrixCol;
use private::PrivateToOwned;
use traits::SumCols;
use {Col, Cols};

impl<'a, D, M: UnsafeMatrixCol<'a, D>> DoubleEndedIterator<Col<D>> for Cols<'a, M> {
    fn next_back(&mut self) -> Option<Col<D>> {
        if self.state == self.stop {
            None
        } else {
            self.stop -= 1;
            Some(unsafe { self.mat.unsafe_col(self.stop) })
        }
    }
}

impl<'a, D, M: UnsafeMatrixCol<'a, D>> Iterator<Col<D>> for Cols<'a, M> {
    fn next(&mut self) -> Option<Col<D>> {
        if self.state == self.stop {
            None
        } else {
            let col = unsafe { self.mat.unsafe_col(self.state) };
            self.state += 1;
            Some(col)
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let exact = self.stop - self.state;
        (exact, Some(exact))
    }
}

impl<
    'a,
    T: BlasAxpy + BlasCopy + One,
    D: PrivateToOwned<T>,
    I: Iterator<Col<D>>,
> SumCols<T> for I {
    fn sum(mut self) -> Option<Col<Vec<T>>> {
        self.next().map(|col| {
            let mut sum = col.0.private_to_owned();

            let n = &to_blasint(sum.len());
            let y = sum.as_mut_ptr();
            let incy = &num::one();
            let axpy = BlasAxpy::axpy(None::<T>);

            for col in self {
                let x = col.0.blas_ptr();
                let incx = &col.0.blas_stride();

                unsafe { axpy(n, &num::one(), x, incx, y, incy) }
            }

            Col(sum)
        })
    }
}
