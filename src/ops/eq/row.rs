use std::iter::order;

use traits::{Iter, Matrix, Slice};
use {Row, RowMut, RowVec};

// Combinations
//
// LHS: Row, RowMut, RowVec
// RHS: Same as LHS
//
// -> 9 implementations

// 1 impls
// Core implementations
impl<'a, 'b, A, B> PartialEq<Row<'a, A>> for Row<'b, B> where B: PartialEq<A> {
    fn eq(&self, rhs: &Row<A>) -> bool {
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
forward!(Row<'a, A> {
    RowMut<'b, B>,
    RowVec<B>,
});

// 3 impls
forward!(RowMut<'a, A> {
    Row<'b, B>,
    RowMut<'b, B>,
    RowVec<B>,
});

// 3 impls
forward!(RowVec<A> {
    Row<'b, B>,
    RowMut<'b, B>,
    RowVec<B>,
});
