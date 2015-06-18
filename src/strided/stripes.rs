use std::{cmp, mem};

use cast::From;

use order::Order;
use traits::Matrix;

macro_rules! next {
    ($n:ident, $split:ident) => {
        fn next(&mut self) -> Option<Self::Item> {
            let n = self.m.$n();

            if n == 0 {
                None
            } else {
                let sz = cmp::min(n, self.size);
                let tmp = mem::replace(&mut self.m, ::Mat::empty());
                let (stripe, left) = tmp.$split(sz);
                self.m = left;
                Some(stripe)
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let exact = usize::from(self.m.$n() / self.size);
            (exact, Some(exact))
        }
    }
}

macro_rules! next_back {
    ($n:ident, $split:ident) => {
        fn next_back(&mut self) -> Option<Self::Item> {
            let n = self.m.$n();

            if n == 0 {
                None
            } else {
                let remainder = n % self.size;
                let sz = if remainder == 0 { self.size } else { remainder };
                let tmp = mem::replace(&mut self.m, ::Mat::empty());
                let (left, stripe) = tmp.$split(n - sz);
                self.m = left;
                Some(stripe)
            }
        }
    }
}

impl<'a, T> DoubleEndedIterator for ::HStripes<'a, T> {
    next_back!(nrows, hsplit_at);
}

impl<'a, T> Iterator for ::HStripes<'a, T> {
    type Item = &'a ::Mat<T, ::order::Row>;

    next!(nrows, hsplit_at);
}

impl<'a, T> DoubleEndedIterator for ::HStripesMut<'a, T> {
    next_back!(nrows, hsplit_at_mut);
}

impl<'a, T> Iterator for ::HStripesMut<'a, T> {
    type Item = &'a mut ::Mat<T, ::order::Row>;

    next!(nrows, hsplit_at_mut);
}

impl<'a, T> DoubleEndedIterator for ::VStripes<'a, T> {
    next_back!(ncols, vsplit_at);
}

impl<'a, T> Iterator for ::VStripes<'a, T> {
    type Item = &'a ::Mat<T, ::order::Col>;

    next!(ncols, vsplit_at);
}

impl<'a, T> DoubleEndedIterator for ::VStripesMut<'a, T> {
    next_back!(ncols, vsplit_at_mut);
}

impl<'a, T> Iterator for ::VStripesMut<'a, T> {
    type Item = &'a mut ::Mat<T, ::order::Col>;

    next!(ncols, vsplit_at_mut);
}
impl<'a, T, O> DoubleEndedIterator for ::strided::HStripes<'a, T, O> where O: Order {
    next_back!(nrows, hsplit_at);
}

impl<'a, T, O> Iterator for ::strided::HStripes<'a, T, O> where O: Order {
    type Item = &'a ::strided::Mat<T, O>;

    next!(nrows, hsplit_at);
}

impl<'a, T, O> DoubleEndedIterator for ::strided::HStripesMut<'a, T, O> where O: Order {
    next_back!(nrows, hsplit_at_mut);
}

impl<'a, T, O> Iterator for ::strided::HStripesMut<'a, T, O> where O: Order {
    type Item = &'a mut ::strided::Mat<T, O>;

    next!(nrows, hsplit_at_mut);
}

impl<'a, T, O> DoubleEndedIterator for ::strided::VStripes<'a, T, O> where O: Order {
    next_back!(ncols, vsplit_at);
}

impl<'a, T, O> Iterator for ::strided::VStripes<'a, T, O> where O: Order {
    type Item = &'a ::strided::Mat<T, O>;

    next!(ncols, vsplit_at);
}

impl<'a, T, O> DoubleEndedIterator for ::strided::VStripesMut<'a, T, O> where O: Order {
    next_back!(ncols, vsplit_at_mut);
}

impl<'a, T, O> Iterator for ::strided::VStripesMut<'a, T, O> where O: Order {
    type Item = &'a mut ::strided::Mat<T, O>;

    next!(ncols, vsplit_at_mut);
}
