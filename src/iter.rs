use std::raw::FatPtr;
use std::{fat_ptr, mem};

use cast::From;

use traits::Matrix;

// All these iterators use the same approach:
//
// - The iterator state is a matrix
// - On each iteration, we use the `split` method to "peel off" a single column/row vector.
// - `split` returns two parts: a vector `v` and a matrix `m`. The state will be updated with the
//   `m` matrix, and the vector `v` will be the returned from the `next` method.
// - We stop when the matrix is empty, i.e. either of its dimensions is zero.
//
// Graphically, a column by column iterator:
//
// Initial state: [0, 1, 2]
//                [3, 4, 5]
//
//            |
// Split: [0] | [1, 2]
//        [3] | [4, 5]
//            |
//        "v"     "m"
//
// New state: [1, 2]
//            [4, 5]
//
// `next` returns: [0]
//                 [3]

macro_rules! next {
    ($n:ident, $split:ident, $len:ident) => {
        fn next(&mut self) -> Option<Self::Item> {
            if self.m.$n() == 0 {
                None
            } else {
                let tmp = mem::replace(&mut self.m, ::strided::Mat::empty());
                let (v, m) = tmp.$split(1);
                self.m = m;
                let FatPtr { data, info: ::strided::mat::Info { $len, .. } } = v.repr();
                let v: *mut ::Vector<T> = fat_ptr::new(FatPtr { data: data, info: $len });
                unsafe {
                    Some(mem::transmute(v))
                }
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let exact = usize::from(self.m.$n());
            (exact, Some(exact))
        }
    }
}

macro_rules! next_back {
    ($n:ident, $split:ident, $len:ident) => {
        fn next_back(&mut self) -> Option<Self::Item> {
            let n = self.m.$n();
            if n == 0 {
                None
            } else {
                let tmp = mem::replace(&mut self.m, ::strided::Mat::empty());
                let (m, v) = tmp.$split(n - 1);
                self.m = m;
                let FatPtr { data, info: ::strided::mat::Info { $len, .. } } = v.repr();
                let v: *mut ::Vector<T> = fat_ptr::new(FatPtr { data: data, info: $len });
                unsafe {
                    Some(mem::transmute(v))
                }
            }
        }
    }
}

macro_rules! strided_next {
    ($n:ident, $split:ident, $len:ident) => {
        fn next(&mut self) -> Option<Self::Item> {
            if self.m.$n() == 0 {
                None
            } else {
                let tmp = mem::replace(&mut self.m, ::strided::Mat::empty());
                let (v, m) = tmp.$split(1);
                self.m = m;
                let FatPtr { data, info: ::strided::mat::Info { $len, stride, .. } } = v.repr();
                let v: *mut ::strided::Vector<T> = fat_ptr::new(FatPtr {
                    data: data,
                    info: ::strided::vector::Info {
                        len: $len,
                        stride: stride,
                    }
                });
                unsafe {
                    Some(mem::transmute(v))
                }
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let exact = usize::from(self.m.$n());
            (exact, Some(exact))
        }
    }
}

macro_rules! strided_next_back {
    ($n:ident, $split:ident, $len:ident) => {
        fn next_back(&mut self) -> Option<Self::Item> {
            let n = self.m.$n();
            if n == 0 {
                None
            } else {
                let tmp = mem::replace(&mut self.m, ::strided::Mat::empty());
                let (m, v) = tmp.$split(n - 1);
                self.m = m;
                let FatPtr { data, info: ::strided::mat::Info { $len, stride, .. } } = v.repr();
                let v: *mut ::strided::Vector<T> = fat_ptr::new(FatPtr {
                    data: data,
                    info: ::strided::vector::Info {
                        len: $len,
                        stride: stride,
                    }
                });
                unsafe {
                    Some(mem::transmute(v))
                }
            }
        }
    }
}

impl<'a, T> DoubleEndedIterator for ::Cols<'a, T, ::order::Col> {
    next_back!(ncols, vsplit_at, nrows);
}

impl<'a, T> DoubleEndedIterator for ::Cols<'a, T, ::order::Row> {
    strided_next_back!(ncols, vsplit_at, nrows);
}

impl<'a, T> DoubleEndedIterator for ::ColsMut<'a, T, ::order::Col> {
    next_back!(ncols, vsplit_at_mut, nrows);
}

impl<'a, T> DoubleEndedIterator for ::ColsMut<'a, T, ::order::Row> {
    strided_next_back!(ncols, vsplit_at_mut, nrows);
}

impl<'a, T> DoubleEndedIterator for ::Rows<'a, T, ::order::Col> {
    strided_next_back!(nrows, hsplit_at, ncols);
}

impl<'a, T> DoubleEndedIterator for ::Rows<'a, T, ::order::Row> {
    next_back!(nrows, hsplit_at, ncols);
}

impl<'a, T> DoubleEndedIterator for ::RowsMut<'a, T, ::order::Col> {
    strided_next_back!(nrows, hsplit_at_mut, ncols);
}

impl<'a, T> DoubleEndedIterator for ::RowsMut<'a, T, ::order::Row> {
    next_back!(nrows, hsplit_at_mut, ncols);
}

impl<'a, T> Iterator for ::Cols<'a, T, ::order::Col> {
    type Item = &'a ::Col<T>;

    next!(ncols, vsplit_at, nrows);
}

impl<'a, T> Iterator for ::Cols<'a, T, ::order::Row> {
    type Item = &'a ::strided::Col<T>;

    strided_next!(ncols, vsplit_at, nrows);
}

impl<'a, T> Iterator for ::ColsMut<'a, T, ::order::Col> {
    type Item = &'a mut ::Col<T>;

    next!(ncols, vsplit_at_mut, nrows);
}

impl<'a, T> Iterator for ::ColsMut<'a, T, ::order::Row> {
    type Item = &'a mut ::strided::Col<T>;

    strided_next!(ncols, vsplit_at_mut, nrows);
}

impl<'a, T> Iterator for ::Rows<'a, T, ::order::Col> {
    type Item = &'a ::strided::Row<T>;

    strided_next!(nrows, hsplit_at, ncols);
}

impl<'a, T> Iterator for ::Rows<'a, T, ::order::Row> {
    type Item = &'a ::Row<T>;

    next!(nrows, hsplit_at, ncols);
}

impl<'a, T> Iterator for ::RowsMut<'a, T, ::order::Col> {
    type Item = &'a mut ::strided::Row<T>;

    strided_next!(nrows, hsplit_at_mut, ncols);
}

impl<'a, T> Iterator for ::RowsMut<'a, T, ::order::Row> {
    type Item = &'a mut ::Row<T>;

    next!(nrows, hsplit_at_mut, ncols);
}
