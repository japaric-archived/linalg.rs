use std::iter::FromIterator;
use std::marker::Unsized;
use std::num::One;
use std::ops::Range;
use std::raw::FatPtr;
use std::{fat_ptr, fmt, mem, ptr, slice};

use cast::From;
use extract::Extract;

use u31::U31;

impl<T> ::Vector<T> {
    pub fn new(slice: &[T]) -> *mut ::Vector<T> {
        fat_ptr::new(FatPtr {
            data: slice.as_ptr() as *mut T,
            info: U31::from(slice.len()).unwrap(),
        })
    }

    pub fn deref(&self) -> *mut ::strided::Vector<T> {
        let FatPtr { data, info: len } = self.repr();

        fat_ptr::new(FatPtr {
            data: data,
            info: ::strided::vector::Info {
                len: len,
                stride: U31::one(),
            }
        })
    }

    pub fn repr(&self) -> FatPtr<T, U31> {
        fat_ptr::repr(self)
    }

    // NOTE Core
    pub fn slice(&self, Range { start, end }: Range<u32>) -> *mut ::Vector<T> {
        unsafe {
            let FatPtr { data, info: len } = self.repr();

            assert!(start <= end);
            assert!(end <= len.u32());

            fat_ptr::new(FatPtr {
                data: data.offset(start as isize),
                info: U31::from(end - start).extract(),
            })
        }
    }

    unsafe fn as_slice(&self) -> *mut [T] {
        let FatPtr { data, info: len } = self.repr();

        slice::from_raw_parts_mut(data, len.usize())
    }
}

impl<T> AsMut<[T]> for ::Vector<T> {
    fn as_mut(&mut self) -> &mut [T] {
        unsafe {
            &mut *self.as_slice()
        }
    }
}

impl<T> AsRef<[T]> for ::Vector<T> {
    fn as_ref(&self) -> &[T] {
        unsafe {
            &*self.as_slice()
        }
    }
}

impl<T> fmt::Debug for ::Vector<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T> Drop for ::Vector<T> {
    fn drop(&mut self) {
        let ptr = self.repr().data;

        if !ptr.is_null() || ptr as usize != mem::POST_DROP_USIZE {
            for x in self.as_ref() {
                unsafe {
                    ptr::read(x);
                }
            }
        }
    }
}

impl<T> FromIterator<T> for Box<::Vector<T>> {
    fn from_iter<I>(it: I) -> Box<::Vector<T>> where I: IntoIterator<Item=T> {
        let mut v: Vec<_> = it.into_iter().collect();
        let data = v.as_mut_ptr();
        let info = U31::from(v.len()).unwrap();

        mem::forget(v);

        unsafe {
            Box::from_raw(fat_ptr::new(FatPtr { data: data, info: info }))
        }
    }
}

impl<T> Unsized for ::Vector<T> {
    type Data = T;
    type Info = U31;

    fn size_of_val(len: U31) -> usize {
        len.usize() * mem::size_of::<T>()
    }
}
