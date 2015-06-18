use blas::Nrm2;
use cast::From;

use traits::Norm;

// NOTE Core
impl<T, O> Norm for ::Mat<T, O> where T: Nrm2 {
    type Output = T::Output;

    fn norm(&self) -> T::Output {
        unsafe {
            let nrm2 = T::nrm2();

            let slice = self.as_ref();
            // FIXME This shouldn't panic
            let ref n = i32::from(slice.len()).unwrap();
            let x = slice.as_ptr();
            let ref incx = 1;

            nrm2(n, x, incx)
        }
    }
}
