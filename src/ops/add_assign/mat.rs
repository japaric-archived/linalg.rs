use std::num::One;
use std::ops::AddAssign;

use blas::{Axpy, Gemm};
use cast::From;

use ops::{Product, Scaled};
use order::Order;
use traits::Matrix;
use u31::U31;

// NOTE Core
impl<'a, T, O> AddAssign<Scaled<&'a ::Mat<T, O>>> for ::Mat<T, O> where T: 'a + Axpy, O: Order {
    fn add_assign(&mut self, rhs: Scaled<&'a ::Mat<T, O>>) {
        unsafe {
            assert_eq!(self.size(), rhs.size());

            let Scaled(ref alpha, x) = rhs;
            let x = x.as_ref();
            let mut len = x.len();
            let mut x = x.as_ptr();
            let mut y = self.repr().data;
            let ref incx = 1;
            let ref incy = 1;

            let axpy = T::axpy();
            loop {
                if let Some(ref n) = i32::from(len) {
                    axpy(n, alpha, x, incx, y, incy);
                    break
                } else {
                    let n = U31::max_value();
                    axpy(&n.i32(), alpha, x, incx, y, incy);
                    len -= n.usize();
                    let n = n.isize();
                    x = x.offset(n);
                    y = y.offset(n);
                }
            }
        }
    }
}

// NOTE Secondary
impl<'a, T, O> AddAssign<&'a ::Mat<T, O>> for ::Mat<T, O> where T: 'a + Axpy + One, O: Order {
    fn add_assign(&mut self, rhs: &'a ::Mat<T, O>) {
        *self += Scaled(T::one(), rhs)
    }
}

// NOTE Core
impl<'a, 'b, T> AddAssign<Scaled<Product<&'a ::ops::Mat<T>, &'b ::ops::Mat<T>>>> for ::ops::Mat<T> where
    T: 'a + 'b + Gemm + One,
{
    fn add_assign(&mut self, rhs: Scaled<Product<&'a ::ops::Mat<T>, &'b ::ops::Mat<T>>>) {
        let Scaled(ref alpha, Product(a, b)) = rhs;
        let c = self;
        let ref beta = T::one();

        ::ops::blas::gemm(alpha, a, b, beta, c)
    }
}

// NOTE Secondary
impl<'a, 'b, T> AddAssign<Product<&'a ::ops::Mat<T>, &'b ::ops::Mat<T>>> for ::ops::Mat<T> where
    T: 'a + 'b + Gemm + One,
{
    fn add_assign(&mut self, rhs: Product<&'a ::ops::Mat<T>, &'b ::ops::Mat<T>>) {
        *self += Scaled(T::one(), rhs);
    }
}

// NB All these "forward impls" shouldn't be necessary, but there is a limitation in the index
// overloading code. See rust-lang/rust#26218
// NOTE Secondary
impl<'a, 'b, T, O> AddAssign<Scaled<Product<&'a ::ops::Mat<T>, &'b ::ops::Mat<T>>>> for ::strided::Mat<T, O> where
    T: 'a + 'b + Gemm + One,
    O: Order,
{
    fn add_assign(&mut self, rhs: Scaled<Product<&'a ::ops::Mat<T>, &'b ::ops::Mat<T>>>) {
        ::ops::Mat::add_assign(&mut **self, rhs)
    }
}
