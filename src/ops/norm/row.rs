use blas::Nrm2;

use ops::norm;
use traits::{Norm, Slice};

use {Row, RowMut, RowVec};

// NOTE Core
impl<'a, T> Norm for Row<'a, T> where T: Nrm2 {
    type Output = T::Output;

    fn norm(&self) -> T::Output {
        norm::strided(self.0)
    }
}

// NOTE Forward
impl<'a, T> Norm for RowMut<'a, T> where T: Nrm2 {
    type Output = T::Output;

    fn norm(&self) -> T::Output {
        self.slice(..).norm()
    }
}

// NOTE Forward
impl<T> Norm for RowVec<T> where T: Nrm2 {
    type Output = T::Output;

    fn norm(&self) -> T::Output {
        self.slice(..).norm()
    }
}
