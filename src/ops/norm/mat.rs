use std::ops::Add;

use blas::Nrm2;
use onezero::Zero;

use ops::norm;
use traits::{Norm, Slice};
use {Mat, SubMat, SubMatMut, Transposed};

// NOTE Core
impl<T, U> Norm for Mat<T> where T: Nrm2<Output=U>, U: Add<Output=U> + Zero {
    type Output = U;

    fn norm(&self) -> U {
        norm::slice(self.as_slice())
    }
}

// NOTE Core
impl<'a, T, U> Norm for SubMat<'a, T> where T: Nrm2<Output=U>, U: Add<Output=U> + Zero {
    type Output = U;

    fn norm(&self) -> U {
        if let Some(slice) = self.as_slice() {
            norm::slice(slice)
        } else {
            unimplemented!();
        }
    }
}

// NOTE Secondary
impl<T, U> Norm for Transposed<Mat<T>> where T: Nrm2<Output=U>, U: Add<Output=U> + Zero {
    type Output = U;

    fn norm(&self) -> U {
        self.0.norm()
    }
}

// NOTE Forward
impl<'a, T, U> Norm for SubMatMut<'a, T> where T: Nrm2<Output=U>, U: Add<Output=U> + Zero {
    type Output = U;

    fn norm(&self) -> U {
        self.slice(..).norm()
    }
}
