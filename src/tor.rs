use std::iter::{FromIterator, IntoIterator, self};
use std::{mem, slice};
use std::ptr::Unique;

use cast::From;
use extract::Extract;
use onezero::{One, Zero};

use Tor;

impl<T> Tor<T> {
    pub fn new(mut elems: Box<[T]>) -> Tor<T> {
        unsafe {
            let data = elems.as_mut_ptr();
            let len = i32::from(elems.len()).unwrap();
            mem::forget(elems);

            Tor {
                data: Unique::new(data),
                len: len,
            }
        }
    }

    pub unsafe fn uninitialized(len: i32) -> Tor<T> {
        debug_assert!(len >= 0);

        let mut v = Vec::with_capacity(usize::from(len).extract());
        let data = v.as_mut_ptr();
        mem::forget(v);

        Tor {
            data: Unique::new(data),
            len: len,
        }
    }

    pub unsafe fn from_elem(len: i32, elem: T) -> Tor<T> where T: Clone {
        debug_assert!(len > 0);

        let mut v: Vec<_> = iter::repeat(elem).take(usize::from(len).extract()).collect();

        let data = v.as_mut_ptr();
        mem::forget(v);

        Tor {
            data: Unique::new(data),
            len: len,
        }
    }

    pub unsafe fn ones(len: i32) -> Tor<T> where T: Clone + One {
        Tor::from_elem(len, T::one())
    }

    pub unsafe fn zeros(len: i32) -> Tor<T> where T: Clone + Zero {
        Tor::from_elem(len, T::zero())
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(*self.data, usize::from(self.len).extract())
        }
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(*self.data, usize::from(self.len).extract())
        }
    }

    pub fn iter(&self) -> slice::Iter<T> {
        unsafe {
            slice::from_raw_parts(*self.data, usize::from(self.len).extract()).iter()
        }
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        unsafe {
            let len = usize::from(self.len).extract();

            slice::from_raw_parts_mut(*self.data, len).iter_mut()
        }
    }

    pub fn len(&self) -> u32 {
        unsafe {
            u32::from(self.len).extract()
        }
    }

    pub unsafe fn raw_index(&self, i: u32) -> *mut T {
        assert!(i < self.len());

        self.data.offset(isize::from(i))
    }
}

impl<T> Clone for Tor<T> where T: Clone {
    fn clone(&self) -> Tor<T> {
        unsafe {
            let mut v = {
                slice::from_raw_parts(*self.data, usize::from(self.len).extract()).to_vec()
            };
            let data = v.as_mut_ptr();
            mem::forget(data);

            Tor {
                data: Unique::new(data),
                ..*self
            }
        }
    }
}

impl<T> Drop for Tor<T> {
    fn drop(&mut self) {
        unsafe {
            let ptr = *self.data;

            if !ptr.is_null() && ptr as usize != mem::POST_DROP_USIZE {
                let len = usize::from(self.len());

                mem::drop(Vec::from_raw_parts(ptr, len, len))
            }
        }
    }
}

impl<T> FromIterator<T> for Tor<T> {
    fn from_iter<I>(it: I) -> Tor<T> where I: IntoIterator<Item=T> {
        unsafe {
            let mut v = it.into_iter().collect::<Vec<_>>();
            let len = i32::from(v.len()).unwrap();
            let data = v.as_mut_ptr();

            mem::forget(v);

            Tor {
                data: Unique::new(data),
                len: len,
            }
        }
    }
}
