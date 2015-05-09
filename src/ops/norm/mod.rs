use blas::Nrm2;
use cast::From;
use extract::Extract;

use Slice;

mod col;
mod mat;
mod row;

fn slice<T>(x: &[T]) -> T::Output where
    T: Nrm2,
{
    unsafe {
        let nrm2 = T::nrm2();

        let mut n = x.len();
        let mut x = x.as_ptr();

        let ref incx = 1;
        let max = usize::from(i32::max_value()).extract();
        let offset = isize::from(i32::max_value());

        while n >= max {
            nrm2(&i32::max_value(), x, incx);

            x = x.offset(offset);
            n -= max;
        }

        nrm2(&i32::from(n).extract(), x, incx)
    }
}

fn strided<T>(x: Slice<T>) -> T::Output where
    T: Nrm2,
{
    unsafe {
        let nrm2 = T::nrm2();

        let ref incx = *x.stride;
        let ref n = x.len;
        let x = *x.data;

        nrm2(n, x, incx)
    }
}
