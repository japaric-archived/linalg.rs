use std::ops::Mul;

use blas::{Dot, Gemm, Gemv};
use onezero::{One, Zero};

use Forward;
use ops::Reduce;
use traits::{Eval, Matrix, Slice};
use {Chain, Col, ColMut, ColVec, Product, Row, RowMut, RowVec, Scaled, Transposed, SubMat};

// Combinations:
//
// LHS: Product<Row, Chain>, Product<Row, Transposed<SubMat>>, Product<Row, SubMat>, Row, &RowMut,
//      &RowVec, Scaled<Product<Row, Chain>>, Scaled<Product<Row, Transposed<SubMat>>>,
//      Scaled<Product<Row, SubMat>>, Scaled<Row>
// RHS: Col, &ColMut, &ColVec, Product<Chain, Col>, Product<Transposed<SubMat>, Col>,
//      Product<SubMat, Col>, Scaled<Col>, Scaled<Product<Chain, Col>>,
//      Scaled<Product<Transposed<SubMat>, Col>>, Scaled<Product<SubMat, Col>>
//
// -> 100 implementations

// 4 impls
// Core implementations
impl<'a, 'b, T> Mul<Col<'a, T>> for Row<'b, T> where T: Dot {
    type Output = T;

    fn mul(self, rhs: Col<T>) -> T {
        unsafe {
            assert_eq_inner_dimensions!(self, rhs);

            let dot = T::dot();
            let n = &self.0.len;
            let x = *self.0.data;
            let incx = &*self.0.stride;
            let y = *rhs.0.data;
            let incy = &*rhs.0.stride;

            dot(n, x, incx, y, incy)
        }
    }
}

impl<'a, 'b, 'c, T> Mul<Product<Chain<'a, T>, Col<'b, T>>> for Row<'c, T> where
    T: Dot + Gemm + Gemv + One + Zero,
{
    type Output = T;

    fn mul(self, rhs: Product<Chain<T>, Col<T>>) -> T {
        unsafe {
            use ops::reduce::RowMulCol::*;

            assert_eq_inner_dimensions!(self, rhs);

            let row = self;
            let Product(chain, col) = rhs;

            let a_mul_b = (row, chain, col).reduce();

            let (row, col) = match a_mul_b {
                RV_C(ref lhs, rhs) => (lhs.slice(..), rhs),
                RV_CV(ref lhs, ref rhs) => (lhs.slice(..), rhs.slice(..)),
                R_CV(lhs, ref rhs) => (lhs, rhs.slice(..)),
            };

            let dot = T::dot();
            let n = &row.0.len;
            let x = *row.0.data;
            let incx = &*row.0.stride;
            let y = *col.0.data;
            let incy = &*col.0.stride;

            dot(n, x, incx, y, incy)
        }
    }
}

// row * mat * col
macro_rules! rmc {
    ($($rhs:ty),+) => {
        $(
            impl<'a, 'b, 'c, T> Mul<$rhs> for Row<'c, T> where
                T: Dot + Gemv + One + Zero,
            {
                type Output = T;

                fn mul(self, rhs: $rhs) -> T {
                    assert_eq_inner_dimensions!(self, rhs);

                    let r = self;
                    let Product(m, c) = rhs;

                    if r.ncols() < c.nrows() {
                        (r * &Product(m, c).eval())
                    } else {
                        &Product(r, m).eval() * c
                    }
                }
            }

         )+
    }
}

rmc!(Product<Transposed<SubMat<'a, T>>, Col<'b, T>>, Product<SubMat<'a, T>, Col<'b, T>>);

// Secondary implementations
// Impl `(row * mat) * col` as `row * (mat * col)`
macro_rules! reassoc {
    ($($lhs:ty { $($bound:ident),* }),+,) => {
        $(
            impl<'a, 'b, 'c, T> Mul<Col<'c, T>> for $lhs where
                T: Dot + Gemv + One + Zero, $(T: $bound),*
            {
                type Output = T;

                fn mul(self, rhs: Col<T>) -> T {
                    self.0 * (self.1 * rhs)
                }
            }

            impl<'a, 'b, 'c, T> Mul<Scaled<Col<'c, T>>> for $lhs where
                T: Dot + Gemv + Mul<Output=T> + One + Zero, $(T: $bound),*
            {
                type Output = T;

                fn mul(self, rhs: Scaled<Col<T>>) -> T {
                    rhs.0 * (self.0 * (self.1 * rhs.1))
                }
            }

            impl<'a, 'b, 'c, T> Mul<Col<'c, T>> for Scaled<$lhs> where
                T: Dot + Gemv + Mul<Output=T> + One + Zero, $(T: $bound),*
            {
                type Output = T;

                fn mul(self, rhs: Col<T>) -> T {
                    self.0 * ((self.1).0 * ((self.1).1 * rhs))
                }
            }

            impl<'a, 'b, 'c, T> Mul<Scaled<Col<'c, T>>> for Scaled<$lhs> where
                T: Dot + Gemv + Mul<Output=T> + One + Zero, $(T: $bound),*
            {
                type Output = T;

                fn mul(self, rhs: Scaled<Col<T>>) -> T {
                    self.0 * rhs.0 * ((self.1).0 * ((self.1).1 * rhs.1))
                }
            }
         )+
    }
}

// 12 impls
reassoc! {
    Product<Row<'a, T>, Chain<'b, T>> { Gemm },
    Product<Row<'a, T>, Transposed<SubMat<'b, T>>> {  },
    Product<Row<'a, T>, SubMat<'b, T>> {  },
}

// Impl `(row * mat) * (mat * col)` as `row * ((mat * mat) * col)`
// 9 impls
impl<'a, 'b, 'c, T, L, R> Mul<Product<R, Col<'a, T>>> for Product<Row<'b, T>, L> where
    L: Matrix<Elem=T> + Mul<R, Output=Chain<'c, T>>,
    R: Matrix<Elem=T>,
    T: Dot + Gemm + Gemv + One + Zero,
{
    type Output = T;

    fn mul(self, rhs: Product<R, Col<'a, T>>) -> T {
        self.0 * Product(self.1 * rhs.0, rhs.1)
    }
}

// 9 impls
impl<'a, 'b, 'c, T, L, R> Mul<Scaled<Product<R, Col<'a, T>>>> for Product<Row<'b, T>, L> where
    L: Matrix<Elem=T> + Mul<R, Output=Chain<'c, T>>,
    R: Matrix<Elem=T>,
    T: Dot + Gemm + Gemv + Mul<Output=T> + One + Zero,
{
    type Output = T;

    fn mul(self, rhs: Scaled<Product<R, Col<'a, T>>>) -> T {
        rhs.0 * (self * rhs.1)
    }
}

// 9 impls
impl<'a, 'b, 'c, T, L, R> Mul<Product<R, Col<'a, T>>> for Scaled<Product<Row<'b, T>, L>> where
    L: Matrix<Elem=T> + Mul<R, Output=Chain<'c, T>>,
    R: Matrix<Elem=T>,
    T: Dot + Gemm + Gemv + Mul<Output=T> + One + Zero,
{
    type Output = T;

    fn mul(self, rhs: Product<R, Col<'a, T>>) -> T {
        self.0 * (self.1 * rhs)
    }
}

// 9 impls
impl<'a, 'b, 'c, T, L, R>
Mul<Scaled<Product<R, Col<'a, T>>>> for Scaled<Product<Row<'b, T>, L>> where
    L: Matrix<Elem=T> + Mul<R, Output=Chain<'c, T>>,
    R: Matrix<Elem=T>,
    T: Dot + Gemm + Gemv + Mul<Output=T> + One + Zero,
{
    type Output = T;

    fn mul(self, rhs: Scaled<Product<R, Col<'a, T>>>) -> T {
        self.0 * rhs.0 * (self.1 * rhs.1)
    }
}

macro_rules! scaled {
    ($lhs:ty, $rhs:ty { $($bound:ident),+ }) => {
        impl<'a, 'b, 'c, T> Mul<$rhs> for Scaled<$lhs> where T: Mul<Output=T>, $(T: $bound),+ {
            type Output = T;

            fn mul(self, rhs: $rhs) -> T {
                self.0 * (self.1 * rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<Scaled<$rhs>> for $lhs where T: Mul<Output=T>, $(T: $bound),+ {
            type Output = T;

            fn mul(self, rhs: Scaled<$rhs>) -> T {
                rhs.0 * (self * rhs.1)
            }
        }

        impl<'a, 'b, 'c, T> Mul<Scaled<$rhs>> for Scaled<$lhs> where
            T: Mul<Output=T>, $(T: $bound),+,
        {
            type Output = T;

            fn mul(self, rhs: Scaled<$rhs>) -> T {
                self.0 * rhs.0 * (self.1 * rhs.1)
            }
        }
    }
}

// 3 impls
scaled!(Row<'a, T>, Col<'b, T> { Dot });

// 3 impls
scaled!(Row<'a, T>, Product<Chain<'b, T>, Col<'c, T>> { Dot, Gemm, Gemv, One, Zero });

// 3 impls
scaled!(Row<'a, T>, Product<Transposed<SubMat<'b, T>>, Col<'c, T>> { Dot, Gemv, One, Zero });

// 3 impls
scaled!(Row<'a, T>, Product<SubMat<'b, T>, Col<'c, T>> { Dot, Gemv, One, Zero });

macro_rules! forward {
    ($lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, 'c, 'd, T> Mul<$rhs> for $lhs where $(T: $bound),+ {
                type Output = T;

                fn mul(self, rhs: $rhs) -> T {
                    self.slice(..) * rhs.slice(..)
                }
            }
         )+
    };
    (scaled $lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, 'c, 'd, T> Mul<$rhs> for $lhs where
                T: Mul<Output=T>, $(T: $bound),+
            {
                type Output = T;

                fn mul(self, rhs: $rhs) -> T {
                    self.slice(..) * rhs.slice(..)
                }
            }
         )+
    };
}

// 2 impls
forward!(Product<Row<'a, T>, Chain<'b, T>> {
    &'c ColMut<'d, T> { Dot, Gemm, Gemv, One, Zero },
    &'c ColVec<T> { Dot, Gemm, Gemv, One, Zero },
});

// 2 impls
forward!(Product<Row<'a, T>, Transposed<SubMat<'b, T>>> {
    &'c ColMut<'d, T> { Dot, Gemv, One, Zero },
    &'c ColVec<T> { Dot, Gemv, One, Zero },
});

// 2 impls
forward!(Product<Row<'a, T>, SubMat<'b, T>> {
    &'c ColMut<'d, T> { Dot, Gemv, One, Zero },
    &'c ColVec<T> { Dot, Gemv, One, Zero },
});

// 2 impls
forward!(Row<'a, T> {
    &'b ColMut<'c, T> { Dot },
    &'b ColVec<T> { Dot },
});

// 10 impls
forward!(&'a RowMut<'b, T> {
    Col<'c, T> { Dot },
    &'c ColMut<'d, T> { Dot },
    &'c ColVec<T> { Dot },
    Product<Chain<'c, T>, Col<'d, T>> { Dot, Gemm, Gemv, One, Zero },
    Product<Transposed<SubMat<'c, T>>, Col<'d, T>> { Dot, Gemv, One, Zero },
    Product<SubMat<'c, T>, Col<'d, T>> { Dot, Gemv, One, Zero },
});
forward!(scaled &'a RowMut<'b, T> {
    Scaled<Col<'c, T>> { Dot },
    Scaled<Product<Chain<'c, T>, Col<'d, T>>> { Dot, Gemm, Gemv, One, Zero },
    Scaled<Product<Transposed<SubMat<'c, T>>, Col<'d, T>>> { Dot, Gemv, One, Zero },
    Scaled<Product<SubMat<'c, T>, Col<'d, T>>> { Dot, Gemv, One, Zero },
});

// 10 impls
forward!(&'a RowVec<T> {
    Col<'b, T> { Dot },
    &'b ColMut<'c, T> { Dot },
    &'b ColVec<T> { Dot },
    Product<Chain<'b, T>, Col<'c, T>> { Dot, Gemm, Gemv, One, Zero },
    Product<Transposed<SubMat<'b, T>>, Col<'c, T>> { Dot, Gemv, One, Zero },
    Product<SubMat<'b, T>, Col<'c, T>> { Dot, Gemv, One, Zero },
});
forward!(scaled &'a RowVec<T> {
    Scaled<Col<'b, T>> { Dot },
    Scaled<Product<Chain<'b, T>, Col<'c, T>>> { Dot, Gemm, Gemv, One, Zero },
    Scaled<Product<Transposed<SubMat<'b, T>>, Col<'c, T>>> { Dot, Gemv, One, Zero },
    Scaled<Product<SubMat<'b, T>, Col<'c, T>>> { Dot, Gemv, One, Zero },
});

// 2 impls
forward!(scaled Scaled<Product<Row<'a, T>, Chain<'b, T>>> {
    &'c ColMut<'d, T> { Dot, Gemm, Gemv, One, Zero },
    &'c ColVec<T> { Dot, Gemm, Gemv, One, Zero },
});

// 2 impls
forward!(scaled Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>> {
    &'c ColMut<'d, T> { Dot, Gemv, One, Zero },
    &'c ColVec<T> { Dot, Gemv, One, Zero },
});

// 2 impls
forward!(scaled Scaled<Product<Row<'a, T>, SubMat<'b, T>>> {
    &'c ColMut<'d, T> { Dot, Gemv, One, Zero },
    &'c ColVec<T> { Dot, Gemv, One, Zero },
});

// 2 impls
forward!(scaled Scaled<Row<'a, T>> {
    &'b ColMut<'c, T> { Dot },
    &'b ColVec<T> { Dot },
});
