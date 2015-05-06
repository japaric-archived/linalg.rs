use std::iter::order;

use traits::{Iter, Matrix, Slice};
use {Col, ColMut, ColVec};

// Combinations
//
// LHS: Col, ColMut, ColVec
// RHS: Same as LHS
//
// -> 9 implementations

// 1 impls
// Core implementation
impl<'a, 'b, A, B> PartialEq<Col<'a, A>> for Col<'b, B> where B: PartialEq<A> {
    fn eq(&self, rhs: &Col<A>) -> bool {
        self.nrows() == rhs.nrows() && order::eq(self.iter(), rhs.iter())
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
    }
}

// 2 impls
forward!(Col<'a, A> {
    ColMut<'b, B>,
    ColVec<B>,
});

// 3 impls
forward!(ColMut<'a, A> {
    Col<'b, B>,
    ColMut<'b, B>,
    ColVec<B>,
});

// 3 impls
forward!(ColVec<A> {
    Col<'a, B>,
    ColMut<'a, B>,
    ColVec<B>,
});
