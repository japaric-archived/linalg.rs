#![doc(hidden)]

mod col;
mod mat;
mod row;

use blas::Scal;
use cast::From;

use u31::U31;

// NOTE Core
fn slice<A, T>(alpha: &A, x: &mut [T]) where T: Scal<A> {
    let mut len = x.len();
    let mut x = x.as_mut_ptr();
    let ref incx = 1;

    unsafe {
        let scal = T::scal();
        loop {
            if let Some(ref n) = i32::from(len) {
                scal(n, alpha, x, incx);
                break
            } else {
                let n = U31::max_value();
                scal(&n.i32(), alpha, x, incx);
                len -= n.usize();
                let n = n.isize();
                x = x.offset(n);
            }
        }
    }
}
