use {Transposed, SubMat, SubMatMut};
use traits::Transpose;
use transposed::{HStripes, HStripesMut, VStripes, VStripesMut};

impl<'a, T> DoubleEndedIterator for HStripes<'a, T> {
    fn next_back(&mut self) -> Option<Transposed<SubMat<'a, T>>> {
        self.0.next_back().map(|v| v.t())
    }
}

impl<'a, T> Iterator for HStripes<'a, T> {
    type Item = Transposed<SubMat<'a, T>>;

    fn next(&mut self) -> Option<Transposed<SubMat<'a, T>>> {
        self.0.next().map(|v| v.t())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for HStripesMut<'a, T> {
    fn next_back(&mut self) -> Option<Transposed<SubMatMut<'a, T>>> {
        self.0.next_back().map(|v| v.t())
    }
}

impl<'a, T> Iterator for HStripesMut<'a, T> {
    type Item = Transposed<SubMatMut<'a, T>>;

    fn next(&mut self) -> Option<Transposed<SubMatMut<'a, T>>> {
        self.0.next().map(|v| v.t())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for VStripes<'a, T> {
    fn next_back(&mut self) -> Option<Transposed<SubMat<'a, T>>> {
        self.0.next_back().map(|v| v.t())
    }
}

impl<'a, T> Iterator for VStripes<'a, T> {
    type Item = Transposed<SubMat<'a, T>>;

    fn next(&mut self) -> Option<Transposed<SubMat<'a, T>>> {
        self.0.next().map(|v| v.t())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for VStripesMut<'a, T> {
    fn next_back(&mut self) -> Option<Transposed<SubMatMut<'a, T>>> {
        self.0.next_back().map(|v| v.t())
    }
}

impl<'a, T> Iterator for VStripesMut<'a, T> {
    type Item = Transposed<SubMatMut<'a, T>>;

    fn next(&mut self) -> Option<Transposed<SubMatMut<'a, T>>> {
        self.0.next().map(|v| v.t())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
