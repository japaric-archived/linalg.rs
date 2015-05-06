use std::ops::{Neg, Sub};

use assign::SubAssign;
use blas::{Axpy, Gemm, Gemv};
use onezero::{One, Zero};

use traits::Transpose;
use {Chain, Product, Row, RowMut, RowVec, Scaled, Transposed, SubMat};

// GEMV
// Combinations:
//
// LHS: RowVec, Scaled<RowVec>
// RHS: Product<Row, Chain>, Product<Row, Transposed<SubMat>>, Product<Row, SubMat>,
//      Scaled<Product<Row, Chain.>>, Scaled<Product<Row, Transposed<SubMat>>>,
//      Scaled<Product<Row, SubMat>>
//
// -> 24 implementations

// Implement `row - row * mat` as `(row^t - (row * mat)^t)^t`
macro_rules! transposed {
    ($lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, T> Sub<$rhs> for $lhs where T: Neg<Output=T>, $(T: $bound),+ {
                type Output = RowVec<T>;

                fn sub(self, rhs: $rhs) -> RowVec<T> {
                    (self.t() - rhs.t()).t()
                }
            }

            impl<'a, 'b, T> Sub<$lhs> for $rhs where T: Neg<Output=T>, $(T: $bound),+ {
                type Output = RowVec<T>;

                fn sub(self, rhs: $lhs) -> RowVec<T> {
                    (self.t() - rhs.t()).t()
                }
            }
         )+
    }
}

// 12 impls
transposed!(RowVec<T> {
    Product<Row<'a, T>, Chain<'b, T>> { Gemm, Gemv, One, Zero },
    Product<Row<'a, T>, Transposed<SubMat<'b, T>>> { Gemv, One },
    Product<Row<'a, T>, SubMat<'b, T>> { Gemv, One },
    Scaled<Product<Row<'a, T>, Chain<'b, T>>> { Gemm, Gemv, One, Zero },
    Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>> { Gemv, One },
    Scaled<Product<Row<'a, T>, SubMat<'b, T>>> { Gemv, One },
});

// 12 impls
transposed!(Scaled<RowVec<T>> {
    Product<Row<'a, T>, Chain<'b, T>> { Gemm, Gemv, One, Zero },
    Product<Row<'a, T>, Transposed<SubMat<'b, T>>> { Gemv, One },
    Product<Row<'a, T>, SubMat<'b, T>> { Gemv, One },
    Scaled<Product<Row<'a, T>, Chain<'b, T>>> { Gemm, Gemv, One, Zero },
    Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>> { Gemv },
    Scaled<Product<Row<'a, T>, SubMat<'b, T>>> { Gemv },
});

// AXPY
// Combinations:
//
// LHS: RowVec
// RHS: &T, T, Row, &RowMut, &RowVec, Scaled<Row>
//
// -> 6 implementations

// 6 impls
assign!(RowVec<T> {
    &'a T { Axpy, One },
    T { Axpy, One },
    Row<'a, T> { Axpy, One },
    &'a RowMut<'b, T> { Axpy, One },
    &'a RowVec<T> { Axpy, One },
    Scaled<Row<'a, T>> { Axpy },
});
