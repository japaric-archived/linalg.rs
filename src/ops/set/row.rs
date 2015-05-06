use blas::Copy;

use ops::{set, self};
use traits::{Matrix, Set, Slice, SliceMut};
use {RowMut, Row, RowVec};

// NOTE Core
impl<'a, T> Set<T> for RowMut<'a, T> where T: Copy {
    fn set(&mut self, value: T) {
        let RowMut(Row(ref mut y)) = *self;
        let ref x = value;

        set::strided(x, y)
    }
}

// NOTE Core
impl<'a, 'b, T> Set<Row<'a, T>> for RowMut<'b, T> where T: Copy {
    fn set(&mut self, rhs: Row<T>) {
        unsafe {
            assert_eq!(self.ncols(), rhs.ncols());

            let RowMut(Row(ref mut y)) = *self;
            let Row(ref x) = rhs;

            ops::copy_strided(x, y)
        }
    }
}

// NOTE Forward
impl<T> Set<T> for RowVec<T> where T: Copy {
    fn set(&mut self, value: T) {
        self.slice_mut(..).set(value)
    }
}

// NOTE Forward
impl<'a, T> Set<Row<'a, T>> for RowVec<T> where T: Copy {
    fn set(&mut self, rhs: Row<T>) {
        self.slice_mut(..).set(rhs)
    }
}

// NOTE Forward
impl<'a, 'b, T> Set<&'a RowMut<'b, T>> for RowVec<T> where T: Copy {
    fn set(&mut self, rhs: &RowMut<T>) {
        self.slice_mut(..).set(rhs.slice(..))
    }
}

// NOTE Forward
impl<'a, T> Set<&'a RowVec<T>> for RowVec<T> where T: Copy {
    fn set(&mut self, rhs: &RowVec<T>) {
        self.slice_mut(..).set(rhs.slice(..))
    }
}
