use std::iter::order;

use traits::{Matrix, MatrixCols, Iter, Slice};
use {Mat, Transposed, SubMat, SubMatMut};

// Combinations
//
// LHS: Mat, Transposed<Mat>, Transposed<SubMat>, Transposed<SubMatMut>, SubMat, SubMatMut
// RHS: Same as LHS
//
// -> 36 implementations

// 2 impls
// Core implementations
impl<'a, 'b, A, B> PartialEq<Transposed<SubMat<'a, A>>> for SubMat<'b, B> where B: PartialEq<A> {
    fn eq(&self, rhs: &Transposed<SubMat<A>>) -> bool {
        let rhs_ = rhs.cols().flat_map(|c| c.iter());

        self.size() == rhs.size() && if let Some(slice) = self.as_slice() {
            order::eq(slice.iter(), rhs_)
        } else {
            order::eq(self.iter(), rhs_)
        }
    }
}

impl<'a, 'b, A, B> PartialEq<SubMat<'a, A>> for SubMat<'b, B> where B: PartialEq<A> {
    fn eq(&self, rhs: &SubMat<A>) -> bool {
        if let (Some(lhs), Some(rhs)) = (self.as_slice(), rhs.as_slice()) {
            lhs == rhs
        } else {
            self.size() == rhs.size() && order::eq(self.iter(), rhs.iter())
        }
    }
}

// 2 impls
// Secondary implementations
impl<'a, 'b, A, B> PartialEq<Transposed<SubMat<'a, A>>> for Transposed<SubMat<'b, B>> where
    B: PartialEq<A>,
{
    fn eq(&self, rhs: &Transposed<SubMat<A>>) -> bool {
        self.0 == rhs.0
    }
}

impl<'a, 'b, A, B> PartialEq<SubMat<'b, B>> for Transposed<SubMat<'a, A>> where
    B: PartialEq<A>,
{
    fn eq(&self, rhs: &SubMat<B>) -> bool {
        *rhs == *self
    }
}

macro_rules! forward {
    ($lhs:ty { $($rhs:ty),+, }) => {
        $(
            impl<'a, 'b, A, B> PartialEq<$rhs> for $lhs where A: PartialEq<B> {
                fn eq(&self, rhs: &$rhs) -> bool {
                    self.slice(..) == rhs.slice(..)
                }
            }
         )+
    };
}

// 6 impls
forward!(Mat<A> {
    Mat<B>,
    Transposed<Mat<B>>,
    Transposed<SubMat<'b, B>>,
    Transposed<SubMatMut<'b, B>>,
    SubMat<'b, B>,
    SubMatMut<'b, B>,
});

// 3 impls
forward!(Transposed<Mat<A>> {
    Transposed<Mat<B>>,
    Transposed<SubMat<'b, B>>,
    Transposed<SubMatMut<'b, B>>,
});

// 3 impls
forward!(Transposed<Mat<B>> {
    Mat<A>,
    SubMat<'a, A>,
    SubMatMut<'a, A>,
});

// 2 impls
forward!(Transposed<SubMat<'a, A>> {
    Transposed<Mat<B>>,
    Transposed<SubMatMut<'b, B>>,
});

// 2 impls
forward!(Transposed<SubMat<'b, B>> {
    Mat<A>,
    SubMatMut<'a, A>,
});

// 3 impls
forward!(Transposed<SubMatMut<'a, A>> {
    Transposed<Mat<B>>,
    Transposed<SubMat<'b, B>>,
    Transposed<SubMatMut<'b, B>>,
});

// 3 impls
forward!(Transposed<SubMatMut<'b, B>> {
    Mat<A>,
    SubMat<'a, A>,
    SubMatMut<'a, A>,
});

// 4 impls
forward!(SubMat<'a, A> {
    Mat<B>,
    Transposed<Mat<B>>,
    Transposed<SubMatMut<'b, B>>,
    SubMatMut<'b, B>,
});

// 6 impls
forward!(SubMatMut<'a, A> {
    Mat<B>,
    Transposed<Mat<B>>,
    Transposed<SubMat<'b, B>>,
    Transposed<SubMatMut<'b, B>>,
    SubMat<'b, B>,
    SubMatMut<'b, B>,
});
