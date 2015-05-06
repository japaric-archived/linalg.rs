mod col;
mod diag;
mod mat;
mod row;

use blas::Copy;
use cast::From;
use extract::Extract;

use Slice;

fn slice<T>(x: &T, y: &mut [T]) where T: Copy {
    unsafe {
        let copy = T::copy();
        let mut n = y.len();
        let incx = &0;
        let mut y = y.as_mut_ptr();
        let incy = &1;

        let max = usize::from(i32::max_value()).extract();
        let offset = isize::from(i32::max_value());

        while n >= max {
            copy(&i32::max_value(), x, incx, y, incy);

            y = y.offset(offset);
            n -= max;
        }

        copy(&i32::from(n).extract(), x, incx, y, incy)
    }
}

fn strided<T>(x: &T, y: &mut Slice<T>) where T: Copy {
    unsafe {
        let copy = T::copy();
        let n = &y.len;
        let incx = &0;
        let incy = &*y.stride;

        let y = *y.data;

        copy(n, x, incx, y, incy)
    }
}
