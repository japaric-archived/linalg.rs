use std::fat_ptr;
use std::num::Zero;
use std::raw::FatPtr;

use cast::From;
use extract::Extract;

use order::Order;
use super::{Iter, IterMut};
use traits::Matrix;
use u31::U31;

macro_rules! next {
    () => {
        fn next(&mut self) -> Option<Self::Item> {
            unsafe {
                let FatPtr { data, info } = self.m.repr();

                match O::order() {
                    ::Order::Col => {
                        if info.ncols == U31::zero() {
                            None
                        } else {
                            let next = &mut *data.offset(self.i.isize());

                            self.i += 1;
                            if self.i == info.nrows {
                                self.i = U31::zero();
                                self.m = &mut *fat_ptr::new(FatPtr {
                                    data: data.offset(info.stride.isize()),
                                    info: ::strided::mat::Info {
                                        _marker: info._marker,
                                        ncols: info.ncols.checked_sub(1).extract(),
                                        nrows: info.nrows,
                                        stride: info.stride,
                                    }
                                })
                            }

                            Some(next)
                        }
                    },
                    ::Order::Row => {
                        if info.nrows == U31::zero() {
                            None
                        } else {
                            let next = &mut *data.offset(self.i.isize());

                            self.i += 1;
                            if self.i == info.ncols {
                                self.i = U31::zero();
                                self.m = &mut *fat_ptr::new(FatPtr {
                                    data: data.offset(info.stride.isize()),
                                    info: ::strided::mat::Info {
                                        _marker: info._marker,
                                        ncols: info.ncols,
                                        nrows: info.nrows.checked_sub(1).extract(),
                                        stride: info.stride,
                                    }
                                })
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
