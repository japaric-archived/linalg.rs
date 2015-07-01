use std::marker::Unsized;
use std::ops::{Deref, DerefMut};
use std::raw::FatPtr;
use std::{fat_ptr, mem};

use order::Order;
use traits::{Matrix, Transpose};
use u31::U31;

#[derive(Clone, Copy)]
pub struct Info {
    pub ncols: U31,
    pub nrows: U31,
    pub order: ::Order,
    pub stride: U31,
}

impl<T> ::ops::Mat<T> {
    fn t_raw(&self) -> *mut ::ops::Mat<T> {
        let FatPtr { data, info } = self.repr();

        fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                ncols: info.nrows,
                nrows: info.ncols,
                order: info.order.t(),
                stride: info.stride,
            }
        })
    }
}

impl<T> Matrix for ::ops::Mat<T> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        self.repr().info.nrows.u32()
    }

    fn ncols(&self) -> u32 {
        self.repr().info.ncols.u32()
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

impl<T> Unsized for ::ops::Mat<T> {
    type Data = T;
    type Info = Info;

    fn size_of_val(info: Info) -> usize {
        mem::size_of::<T>() * info.stride.usize() * match info.order {
            ::Order::Col => info.ncols.usize(),
            ::Order::Row => info.nrows.usize(),
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
        let FatPtr { data, info } = self.repr();

        fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                ncols: info.ncols,
                nrows: info.nrows,
                order: O::order(),
                stride: info.stride,
            }
        })
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
