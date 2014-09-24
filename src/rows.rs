use std::num::{One, mod};

use blas::axpy::BlasAxpy;
use blas::copy::BlasCopy;
use blas::{BlasPtr, BlasStride, to_blasint};
use notsafe::UnsafeMatrixRow;
use private::PrivateToOwned;
use traits::SumRows;
use {Row, Rows};

impl<'a, D, M: UnsafeMatrixRow<'a, D>> DoubleEndedIterator<Row<D>> for Rows<'a, M> {
    fn next_back(&mut self) -> Option<Row<D>> {
        if self.state == self.stop {
            None
        } else {
            self.stop -= 1;
            Some(unsafe { self.mat.unsafe_row(self.stop) })
        }
    }
}

impl<'a, D, M: UnsafeMatrixRow<'a, D>> Iterator<Row<D>> for Rows<'a, M> {
    fn next(&mut self) -> Option<Row<D>> {
        if self.state == self.stop {
            None
        } else {
            let  row = unsafe { self.mat.unsafe_row(self.state) };
            self.state += 1;
            Some(row)
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
    I: Iterator<Row<D>>,
> SumRows<T> for I {
    fn sum(mut self) -> Option<Row<Vec<T>>> {
        self.next().map(|row| {
            let mut sum = row.0.private_to_owned();

            let n = &to_blasint(sum.len());
            let y = sum.as_mut_ptr();
            let incy = &num::one();
            let axpy = BlasAxpy::axpy(None::<T>);

            for row in self {
                let x = row.0.blas_ptr();
                let incx = &row.0.blas_stride();

                unsafe { axpy(n, &num::one(), x, incx, y, incy) }
            }

            Row(sum)
        })
    }
}
