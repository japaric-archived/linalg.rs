use std::mem;

use cast::From;

use traits::Matrix;

macro_rules! next {
    ($n:ident, $split:ident, $len:ident) => {
        fn next(&mut self) -> Option<Self::Item> {
            unsafe {
                let n = self.m.$n();
                if n == 0 {
                    None
                } else {
                    let tmp = mem::replace(&mut self.m, ::strided::Mat::empty());
                    let (slice, left) = tmp.$split(1);
                    self.m = left;
                    let ::strided::raw::Mat { data, $len, .. } = slice.repr();
                    Some(mem::transmute(::raw::Slice { data: data, len: $len }))
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
            unsafe {
                let n = self.m.$n();
                if n == 0 {
                    None
                } else {
                    let tmp = mem::replace(&mut self.m, ::strided::Mat::empty());
                    let (left, slice) = tmp.$split(n - 1);
                    self.m = left;
                    let ::strided::raw::Mat { data, $len, .. } = slice.repr();
                    Some(mem::transmute(::raw::Slice { data: data, len: $len }))
                }
            }
        }
    }
}

macro_rules! strided_next {
    ($n:ident, $split:ident, $len:ident) => {
        fn next(&mut self) -> Option<Self::Item> {
            unsafe {
                let n = self.m.$n();
                if n == 0 {
                    None
                } else {
                    let tmp = mem::replace(&mut self.m, ::strided::Mat::empty());
                    let (slice, left) = tmp.$split(1);
                    self.m = left;
                    let ::strided::raw::Mat { data, $len, stride, .. } = slice.repr();
                    Some(mem::transmute(::strided::raw::Slice {
                        data: data,
                        len: $len,
                        stride: stride,
                    }))
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
            unsafe {
                let n = self.m.$n();
                if n == 0 {
                    None
                } else {
                    let tmp = mem::replace(&mut self.m, ::strided::Mat::empty());
                    let (left, slice) = tmp.$split(n - 1);
                    self.m = left;
                    let ::strided::raw::Mat { data, $len, stride, .. } = slice.repr();
                    Some(mem::transmute(::strided::raw::Slice {
                        data: data,
                        len: $len,
                        stride: stride
                    }))
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
