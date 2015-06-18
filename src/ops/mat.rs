use std::mem;
use std::ops::{Deref, DerefMut};

use order::Order;
use traits::{Matrix, Transpose};

impl<T> ::ops::Mat<T> {
    fn t_raw(&self) -> *mut ::ops::Mat<T> {
        unsafe {
            let ::ops::raw::Mat { data, nrows, ncols, stride, order } = self.repr();

            mem::transmute(::ops::raw::Mat {
                data: data,
                ncols: nrows,
                nrows: ncols,
                order: order.t(),
                stride: stride,
            })
        }
    }
}

impl<T> Matrix for ::ops::Mat<T> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        self.repr().nrows.u32()
    }

    fn ncols(&self) -> u32 {
        self.repr().ncols.u32()
    }
}

impl<'a, T> Transpose for &'a ::ops::Mat<T> {
    type Output = &'a ::ops::Mat<T>;

    fn t(self) -> &'a ::ops::Mat<T> {
        unsafe {
            &*self.t_raw()
        }
    }
}

impl ::Order {
    fn t(&self) -> ::Order {
        match *self {
            ::Order::Col => ::Order::Row,
            ::Order::Row => ::Order::Col,
        }
    }
}

impl<T, O> ::strided::Mat<T, O> where O: Order {
    // NOTE Core
    fn deref_raw(&self) -> *mut ::ops::Mat<T> {
        unsafe {
            let ::strided::raw::Mat { data, nrows, ncols, stride, .. } = self.repr();

            mem::transmute(::ops::raw::Mat {
                data: data,
                ncols: ncols,
                nrows: nrows,
                order: O::order(),
                stride: stride,
            })
        }
    }
}

impl<T, O> Deref for ::strided::Mat<T, O> where O: Order {
    type Target = ::ops::Mat<T>;

    fn deref(&self) -> &::ops::Mat<T> {
        unsafe {
            &*self.deref_raw()
        }
    }
}

impl<T, O> DerefMut for ::strided::Mat<T, O> where O: Order {
    fn deref_mut(&mut self) -> &mut ::ops::Mat<T> {
        unsafe {
            &mut *self.deref_raw()
        }
    }
}
