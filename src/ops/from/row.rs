use blas::Copy;

use ops::from;
use traits::Slice as _0;
use {Row, RowMut, RowVec, Slice, Tor};

// NOTE Core
impl<'a, T> From<Row<'a, T>> for RowVec<T> where T: Copy {
    fn from(input: Row<T>) -> RowVec<T> {
        unsafe {
            RowVec(from::strided_to_tor(&input.0))
        }
    }
}

// NOTE Core
impl<'a, T> From<&'a [T]> for Row<'a, T> {
    fn from(slice: &[T]) -> Row<T> {
        unsafe {
            use cast::From as _0;

            let data = slice.as_ptr();
            let len = i32::from_(slice.len()).unwrap();

            Row(Slice::new(data as *mut T, len, 1))
        }
    }
}

// NOTE Core
impl<T> From<Box<[T]>> for RowVec<T> {
    fn from(slice: Box<[T]>) -> RowVec<T> {
        RowVec(Tor::new(slice))
    }
}

// NOTE Forward
impl<T> From<Vec<T>> for RowVec<T> {
    fn from(v: Vec<T>) -> RowVec<T> {
        RowVec::from(v.into_boxed_slice())
    }
}

// NOTE Forward
impl<'a, T> From<&'a mut [T]> for RowMut<'a, T> {
    fn from(slice: &mut [T]) -> RowMut<T> {
        RowMut(Row::from(&*slice))
    }
}

// NOTE Forward
impl<'a, 'b, T> From<&'a RowMut<'b, T>> for RowVec<T> where T: Copy {
    fn from(input: &RowMut<T>) -> RowVec<T> {
        RowVec::from(input.slice(..))
    }
}
