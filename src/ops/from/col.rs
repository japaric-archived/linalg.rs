use ops::from;

use blas::Copy;

use traits::Slice as _0;
use {Col, ColMut, ColVec, Slice, Tor};

// NOTE Core
impl<'a, T> From<Col<'a, T>> for ColVec<T> where T: Copy {
    fn from(input: Col<T>) -> ColVec<T> {
        unsafe {
            ColVec(from::strided_to_tor(&input.0))
        }
    }
}

// NOTE Core
impl<'a, T> From<&'a [T]> for Col<'a, T> {
    fn from(slice: &[T]) -> Col<T> {
        unsafe {
            use cast::From as _0;

            let data = slice.as_ptr();
            let len = i32::from_(slice.len()).unwrap();

            Col(Slice::new(data as *mut T, len, 1))
        }
    }
}

// NOTE Core
impl<T> From<Box<[T]>> for ColVec<T> {
    fn from(slice: Box<[T]>) -> ColVec<T> {
        ColVec(Tor::new(slice))
    }
}

// NOTE Forward
impl<T> From<Vec<T>> for ColVec<T> {
    fn from(v: Vec<T>) -> ColVec<T> {
        ColVec::from(v.into_boxed_slice())
    }
}

// NOTE Forward
impl<'a, T> From<&'a mut [T]> for ColMut<'a, T> {
    fn from(slice: &mut [T]) -> ColMut<T> {
        ColMut(Col::from(&*slice))
    }
}

// NOTE Forward
impl<'a, 'b, T> From<&'a ColMut<'b, T>> for ColVec<T> where T: Copy {
    fn from(input: &ColMut<T>) -> ColVec<T> {
        ColVec::from(input.slice(..))
    }
}
