use std::cmp;

use cast::From;
use extract::Extract;

use traits::Matrix;
use {HStripes, HStripesMut, VStripes, VStripesMut, SubMat, SubMatMut};

impl<'a, T> DoubleEndedIterator for HStripes<'a, T> {
    fn next_back(&mut self) -> Option<SubMat<'a, T>> {
        unsafe {
            if self.mat.nrows == 0 {
                None
            } else {
                let size = cmp::min(self.size, self.mat.nrows);
                let (left, stripe) = self.mat.unsafe_hsplit_at(self.mat.nrows - size);

                self.mat = left;

                Some(stripe)
            }
        }
    }
}

impl<'a, T> Iterator for HStripes<'a, T> {
    type Item = SubMat<'a, T>;

    fn next(&mut self) -> Option<SubMat<'a, T>> {
        unsafe {
            if self.mat.nrows == 0 {
                None
            } else {
                let size = cmp::min(self.size, self.mat.nrows);
                let (stripe, left) = self.mat.unsafe_hsplit_at(size);
                self.mat = left;

                Some(stripe)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        unsafe {
            let exact = usize::from({
                self.mat.nrows / self.size + if self.mat.nrows % self.size == 0 { 0 } else { 1 }
            }).extract();

            (exact, Some(exact))
        }
    }
}

impl<'a, T> DoubleEndedIterator for HStripesMut<'a, T> {
    fn next_back(&mut self) -> Option<SubMatMut<'a, T>> {
        self.0.next_back().map(|v| SubMatMut(v))
    }
}

impl<'a, T> Iterator for HStripesMut<'a, T> {
    type Item = SubMatMut<'a, T>;

    fn next(&mut self) -> Option<SubMatMut<'a, T>> {
        self.0.next().map(|v| SubMatMut(v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for VStripes<'a, T> {
    fn next_back(&mut self) -> Option<SubMat<'a, T>> {
        unsafe {
            if self.mat.ncols == 0 {
                None
            } else {
                let size = cmp::min(self.size, self.mat.ncols);
                let (left, stripe) = self.mat.unsafe_vsplit_at(self.mat.ncols - size);

                self.mat = left;

                Some(stripe)
            }
        }
    }
}

impl<'a, T> Iterator for VStripes<'a, T> {
    type Item = SubMat<'a, T>;

    fn next(&mut self) -> Option<SubMat<'a, T>> {
        unsafe {
            if self.mat.ncols == 0 {
                None
            } else {
                let size = cmp::min(self.size, self.mat.ncols);
                let (stripe, left) = self.mat.unsafe_vsplit_at(size);
                self.mat = left;

                Some(stripe)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        unsafe {
            let exact = usize::from({
                self.mat.ncols / self.size + if self.mat.ncols % self.size == 0 { 0 } else { 1 }
            }).extract();

            (exact, Some(exact))
        }
    }
}

impl<'a, T> DoubleEndedIterator for VStripesMut<'a, T> {
    fn next_back(&mut self) -> Option<SubMatMut<'a, T>> {
        self.0.next_back().map(|v| SubMatMut(v))
    }
}

impl<'a, T> Iterator for VStripesMut<'a, T> {
    type Item = SubMatMut<'a, T>;

    fn next(&mut self) -> Option<SubMatMut<'a, T>> {
        self.0.next().map(|v| SubMatMut(v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
