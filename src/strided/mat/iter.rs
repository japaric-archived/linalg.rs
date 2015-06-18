use std::mem;
use std::num::Zero;

use cast::From;
use core::nonzero::NonZero;
use extract::Extract;

use order::Order;
use super::{Iter, IterMut};
use traits::Matrix;
use u31::U31;

macro_rules! next {
    () => {
        fn next(&mut self) -> Option<Self::Item> {
            unsafe {
                let ::strided::raw::Mat { data, nrows, ncols, stride, marker } = self.m.repr();

                match O::order() {
                    ::Order::Col => {
                        if ncols == U31::zero() {
                            None
                        } else {
                            let next = &mut *data.offset(self.i.isize());

                            self.i += 1;
                            if self.i == nrows {
                                self.i = U31::zero();
                                self.m = mem::transmute(::strided::raw::Mat {
                                    data: NonZero::new(data.offset(stride.isize())),
                                    marker: marker,
                                    ncols: ncols.checked_sub(1).extract(),
                                    nrows: nrows,
                                    stride: stride,
                                });
                            }

                            Some(next)
                        }
                    },
                    ::Order::Row => {
                        if nrows == U31::zero() {
                            None
                        } else {
                            let next = &mut *data.offset(self.i.isize());

                            self.i += 1;
                            if self.i == ncols {
                                self.i = U31::zero();
                                self.m = mem::transmute(::strided::raw::Mat {
                                    data: NonZero::new(data.offset(stride.isize())),
                                    marker: marker,
                                    ncols: ncols,
                                    nrows: nrows.checked_sub(1).extract(),
                                    stride: stride,
                                });
                            }

                            Some(next)
                        }
                    },
                }
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let (nrows, ncols) = self.m.size();
            let exact = usize::from(nrows) * usize::from(ncols) - self.i.usize();
            (exact, Some(exact))
        }
    }
}

impl<'a, T, O> Iterator for Iter<'a, T, O> where O: Order {
    type Item = &'a T;

    next!();
}

impl<'a, T, O> Iterator for IterMut<'a, T, O> where O: Order {
    type Item = &'a mut T;

    next!();
}
