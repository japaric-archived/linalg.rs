use std::ops::Add;

use assign::AddAssign;
use blas::{Axpy, Gemm, Gemv};
use complex::Complex;
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
// and the reverse operation
//
// -> 24 implementations

// Implement `row + row * mat` as `(row^t + (row * mat)^t)^t`
macro_rules! transposed {
    ($lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, T> Add<$rhs> for $lhs where $(T: $bound),+ {
                type Output = RowVec<T>;

                fn add(self, rhs: $rhs) -> RowVec<T> {
                    (self.t() + rhs.t()).t()
                }
            }

            impl<'a, 'b, T> Add<$lhs> for $rhs where $(T: $bound),+ {
                type Output = RowVec<T>;

                fn add(self, rhs: $lhs) -> RowVec<T> {
                    (self.t() + rhs.t()).t()
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
// and the reverse operation
//
// -> 12 implementations

// 10 impls
assign!(half RowVec<T> {
    &'a T { Axpy, One },
    T { Axpy, One },
});

assign!(RowVec<T> {
    Row<'a, T> { Axpy, One },
    &'a RowMut<'b, T> { Axpy, One },
    &'a RowVec<T> { Axpy, One },
    Scaled<Row<'a, T>> { Axpy },
});

macro_rules! scalar {
    ($($t:ty),+) => {
        $(
            impl<'a> Add<RowVec<$t>> for &'a $t {
                type Output = RowVec<$t>;

                fn add(self, mut rhs: RowVec<$t>) -> RowVec<$t> {
                    rhs.add_assign(self);
                    rhs
                }
            }

            impl<'a> Add<RowVec<$t>> for $t {
                type Output = RowVec<$t>;

                fn add(self, mut rhs: RowVec<$t>) -> RowVec<$t> {
                    rhs.add_assign(self);
                    rhs
                }
            }
         )+
    };
}

// 2 impls
scalar!(f32, f64, Complex<f32>, Complex<f64>);
