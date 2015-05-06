use blas::Copy;

use ops::{set, self};
use traits::{Matrix, Set, Slice, SliceMut};
use {ColMut, Col, ColVec};

// NOTE Core
impl<'a, T> Set<T> for ColMut<'a, T> where T: Copy {
    fn set(&mut self, value: T) {
        let ColMut(Col(ref mut y)) = *self;
        let ref x = value;

        set::strided(x, y)
    }
}

// NOTE Core
impl<'a, 'b, T> Set<Col<'a, T>> for ColMut<'b, T> where T: Copy {
    fn set(&mut self, rhs: Col<T>) {
        unsafe {
            assert_eq!(self.nrows(), rhs.nrows());

            let ColMut(Col(ref mut y)) = *self;
            let Col(ref x) = rhs;

            ops::copy_strided(x, y)
        }
    }
}

// NOTE Forward
impl<T> Set<T> for ColVec<T> where T: Copy {
    fn set(&mut self, value: T) {
        self.slice_mut(..).set(value)
    }
}

// NOTE Forward
impl<'a, T> Set<Col<'a, T>> for ColVec<T> where T: Copy {
    fn set(&mut self, rhs: Col<T>) {
        self.slice_mut(..).set(rhs)
    }
}

// NOTE Forward
impl<'a, 'b, T> Set<&'a ColMut<'b, T>> for ColVec<T> where T: Copy {
    fn set(&mut self, rhs: &ColMut<T>) {
        self.slice_mut(..).set(rhs.slice(..))
    }
}

// NOTE Forward
impl<'a, T> Set<&'a ColVec<T>> for ColVec<T> where T: Copy {
    fn set(&mut self, rhs: &ColVec<T>) {
        self.slice_mut(..).set(rhs.slice(..))
    }
}
