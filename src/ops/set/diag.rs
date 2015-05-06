use blas::Copy;

use ops::set;
use traits::Set;
use {DiagMut, Diag};

impl<'a, T> Set<T> for DiagMut<'a, T> where T: Copy {
    fn set(&mut self, value: T) {
        let DiagMut(Diag(ref mut y)) = *self;
        let ref x = value;

        set::strided(x, y)
    }
}
