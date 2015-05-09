use blas::Nrm2;

use ops::norm;
use traits::{Norm, Slice};

use {Col, ColMut, ColVec};

// NOTE Core
impl<'a, T> Norm for Col<'a, T> where T: Nrm2 {
    type Output = T::Output;

    fn norm(&self) -> T::Output {
        norm::strided(self.0)
    }
}

// NOTE Forward
impl<'a, T> Norm for ColMut<'a, T> where T: Nrm2 {
    type Output = T::Output;

    fn norm(&self) -> T::Output {
        self.slice(..).norm()
    }
}

// NOTE Forward
impl<T> Norm for ColVec<T> where T: Nrm2 {
    type Output = T::Output;

    fn norm(&self) -> T::Output {
        self.slice(..).norm()
    }
}
