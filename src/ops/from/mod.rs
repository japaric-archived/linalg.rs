mod col;
mod mat;
mod row;

use std::mem;
use std::ptr::Unique;

use blas::Copy;
use cast::From;
use extract::Extract;

use {Slice, Tor};

unsafe fn slice<T>(input: &[T]) -> Unique<T> where T: Copy {
    let mut n = input.len();
    let mut v = Vec::with_capacity(n);

    let copy = T::copy();
    let mut x = input.as_ptr();
    let incx = &1;
    let mut y = v.as_mut_ptr();
    let incy = &1;

    let max = usize::from(i32::max_value()).extract();
    let offset = isize::from(i32::max_value());

    while n >= max {
        copy(&i32::max_value(), x, incx, y, incy);

        x = x.offset(offset);
        y = y.offset(offset);
        n -= max;
    }

    copy(&i32::from(n).extract(), x, incx, y, incy);

    let data = v.as_mut_ptr();
    mem::forget(v);
    Unique::new(data)
}

unsafe fn strided_to_tor<T>(input: &Slice<T>) -> Tor<T> where T: Copy {
    let output = Tor::uninitialized(input.len);

    {
        let copy = T::copy();
        let n = &input.len;
        let x: *const T = *input.data;
        let incx = &*input.stride;
        let y = *output.data;
        let incy = &1;

        copy(n, x, incx, y, incy);
    }

    output
}
