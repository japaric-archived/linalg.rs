use std::ops::Add;

use blas::Nrm2;
use onezero::Zero;

use ops::norm;
use traits::Norm;
use {Mat, Transposed};

// NOTE Core
impl<T, U> Norm for Mat<T> where T: Nrm2<Output=U>, U: Add<Output=U> + Zero {
    type Output = U;

    fn norm(&self) -> U {
        norm::slice(self.as_slice())
    }
}

// NOTE Secondary
impl<T, U> Norm for Transposed<Mat<T>> where T: Nrm2<Output=U>, U: Add<Output=U> + Zero {
    type Output = U;

    fn norm(&self) -> U {
        self.0.norm()
    }
}
