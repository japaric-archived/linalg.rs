use std::ops::Mul;

use Forward;
use traits::{Matrix, Slice};
use {Chain, Mat, Product, Row, RowMut, RowVec, Scaled, Transposed, SubMat, SubMatMut};

// Combinations:
//
// LHS: Product<Row, Chain>, Product<Row, Transposed<SubMat>>, Product<Row, SubMat>, Row, &RowMut,
//      &RowVec, Scaled<Product<Row, Chain>>, Scaled<Product<Row, Transposed<SubMat>>>,
//      Scaled<Product<Row, SubMat>>, Scaled<Row>
// RHS: Chain, &Mat, Scaled<Chain>, Scaled<Transposed<SubMat>>, Scaled<SubMat>, &Transposed<Mat>,
//      Transposed<SubMat>, &Transposed<SubMatMut>, SubMat, &SubMatMut
//
// -> 100 implementations

macro_rules! mul {
    ($lhs:ty, $rhs:ty) => {
        // Core implementations
        impl<'a, 'b, T> Mul<$rhs> for $lhs {
            type Output = Product<$lhs, $rhs>;

            fn mul(self, rhs: $rhs) -> Product<$lhs, $rhs> {
                assert_eq_inner_dimensions!(self, rhs);

                Product(self, rhs)
            }
        }

        // Secondary implementations
        impl<'a, 'b, T> Mul<Scaled<$rhs>> for $lhs {
            type Output = Scaled<Product<$lhs, $rhs>>;

            fn mul(self, rhs: Scaled<$rhs>) -> Scaled<Product<$lhs, $rhs>> {
                Scaled(rhs.0, self * rhs.1)
            }
        }

        impl<'a, 'b, T> Mul<$rhs> for Scaled<$lhs> {
            type Output = Scaled<Product<$lhs, $rhs>>;

            fn mul(self, rhs: $rhs) -> Scaled<Product<$lhs, $rhs>> {
                Scaled(self.0, self.1 * rhs)
            }
        }

        impl<'a, 'b, T> Mul<Scaled<$rhs>> for Scaled<$lhs> where T: Mul<Output=T> {
            type Output = Scaled<Product<$lhs, $rhs>>;

            fn mul(self, rhs: Scaled<$rhs>) -> Scaled<Product<$lhs, $rhs>> {
                Scaled(self.0 * rhs.0, self.1 * rhs.1)
            }
        }
    };
}

// 12 impls
mul!(Row<'a, T>, Chain<'b, T>);
mul!(Row<'a, T>, Transposed<SubMat<'b, T>>);
mul!(Row<'a, T>, SubMat<'b, T>);

macro_rules! product {
    ($lhs:ty { $($rhs:ty),+ }) => {
        $(
            // Secondary implementations
            impl<'a, 'b, T> Mul<$rhs> for Product<Row<'b, T>, $lhs> {
                type Output = Product<Row<'b, T>, Chain<'a, T>>;

                fn mul(self, rhs: $rhs) -> Product<Row<'b, T>, Chain<'a, T>> {
                    Product(self.0, self.1 * rhs)
                }
            }

            impl<'a, 'b, T> Mul<Scaled<$rhs>> for Product<Row<'b, T>, $lhs> {
                type Output = Scaled<Product<Row<'b, T>, Chain<'a, T>>>;

                fn mul(self, rhs: Scaled<$rhs>) -> Scaled<Product<Row<'b, T>, Chain<'a, T>>> {
                    Scaled(rhs.0, self * rhs.1)
                }
            }

            impl<'a, 'b, T> Mul<$rhs> for Scaled<Product<Row<'b, T>, $lhs>> {
                type Output = Scaled<Product<Row<'b, T>, Chain<'a, T>>>;

                fn mul(self, rhs: $rhs) -> Scaled<Product<Row<'b, T>, Chain<'a, T>>> {
                    Scaled(self.0, self.1 * rhs)
                }
            }

            impl<'a, 'b, T> Mul<Scaled<$rhs>> for Scaled<Product<Row<'b, T>, $lhs>> where
                T: Mul<Output=T>,
            {
                type Output = Scaled<Product<Row<'b, T>, Chain<'a, T>>>;

                fn mul(self, rhs: Scaled<$rhs>) -> Scaled<Product<Row<'b, T>, Chain<'a, T>>> {
                    Scaled(rhs.0 * self.0, self.1 * rhs.1)
                }
            }
         )+
    }
}

// 12 impls
product!(Chain<'a, T> { Chain<'a, T>, Transposed<SubMat<'a, T>>, SubMat<'a, T> });

// 12 impls
product!(Transposed<SubMat<'a, T>> { Chain<'a, T>, Transposed<SubMat<'a, T>>, SubMat<'a, T> });

// 12 impls
product!(SubMat<'a, T> { Chain<'a, T>, Transposed<SubMat<'a, T>>, SubMat<'a, T> });

macro_rules! forward {
    ($lhs:ty { $($rhs:ty => $output:ty),+, }) => {
        $(
            impl<'a, 'b, 'c, 'd, T> Mul<$rhs> for $lhs {
                type Output = $output;

                fn mul(self, rhs: $rhs) -> $output {
                    self.slice(..) * rhs.slice(..)
                }
            }
         )+
    }
}

// 4 impls
forward!(Product<Row<'a, T>, Chain<'b, T>> {
    &'b Mat<T>
        => Product<Row<'a, T>, Chain<'b, T>>,

    &'b Transposed<Mat<T>>
        => Product<Row<'a, T>, Chain<'b, T>>,

    &'b Transposed<SubMatMut<'c, T>>
        => Product<Row<'a, T>, Chain<'b, T>>,

    &'b SubMatMut<'c, T>
        => Product<Row<'a, T>, Chain<'b, T>>,
});

// 4 impls
forward!(Product<Row<'a, T>, Transposed<SubMat<'b, T>>> {
    &'b Mat<T>
        => Product<Row<'a, T>, Chain<'b, T>>,

    &'b Transposed<Mat<T>>
        => Product<Row<'a, T>, Chain<'b, T>>,

    &'b Transposed<SubMatMut<'c, T>>
        => Product<Row<'a, T>, Chain<'b, T>>,

    &'b SubMatMut<'c, T>
        => Product<Row<'a, T>, Chain<'b, T>>,
});

// 4 impls
forward!(Product<Row<'a, T>, SubMat<'b, T>> {
    &'b Mat<T>
        => Product<Row<'a, T>, Chain<'b, T>>,

    &'b Transposed<Mat<T>>
        => Product<Row<'a, T>, Chain<'b, T>>,

    &'b Transposed<SubMatMut<'c, T>>
        => Product<Row<'a, T>, Chain<'b, T>>,

    &'b SubMatMut<'c, T>
        => Product<Row<'a, T>, Chain<'b, T>>,
});

// 4 impls
forward!(Row<'a, T> {
    &'b Mat<T>
        => Product<Row<'a, T>, SubMat<'b, T>>,

    &'b Transposed<Mat<T>>
        => Product<Row<'a, T>, Transposed<SubMat<'b, T>>>,

    &'b Transposed<SubMatMut<'c, T>>
        => Product<Row<'a, T>, Transposed<SubMat<'b, T>>>,

    &'b SubMatMut<'c, T>
        => Product<Row<'a, T>, SubMat<'b, T>>,
});

// 10 impls
forward!(&'a RowMut<'b, T> {
    Chain<'c, T>
        => Product<Row<'a, T>, Chain<'c, T>>,

    &'c Mat<T>
        => Product<Row<'a, T>, SubMat<'c, T>>,

    Scaled<Chain<'c, T>>
        => Scaled<Product<Row<'a, T>, Chain<'c, T>>>,

    Scaled<Transposed<SubMat<'c, T>>>
        => Scaled<Product<Row<'a, T>, Transposed<SubMat<'c, T>>>>,

    Scaled<SubMat<'c, T>>
        => Scaled<Product<Row<'a, T>, SubMat<'c, T>>>,

    &'c Transposed<Mat<T>>
        => Product<Row<'a, T>, Transposed<SubMat<'c, T>>>,

    Transposed<SubMat<'c, T>>
        => Product<Row<'a, T>, Transposed<SubMat<'c, T>>>,

    &'c Transposed<SubMatMut<'d, T>>
        => Product<Row<'a, T>, Transposed<SubMat<'c, T>>>,

    SubMat<'c, T>
        => Product<Row<'a, T>, SubMat<'c, T>>,

    &'c SubMatMut<'d, T>
        => Product<Row<'a, T>, SubMat<'c, T>>,
});

// 10 impls
forward!(&'a RowVec<T> {
    Chain<'b, T>
        => Product<Row<'a, T>, Chain<'b, T>>,

    &'b Mat<T>
        => Product<Row<'a, T>, SubMat<'b, T>>,

    Scaled<Chain<'b, T>>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,

    Scaled<Transposed<SubMat<'b, T>>>
        => Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>>,

    Scaled<SubMat<'b, T>>
        => Scaled<Product<Row<'a, T>, SubMat<'b, T>>>,

    &'b Transposed<Mat<T>>
        => Product<Row<'a, T>, Transposed<SubMat<'b, T>>>,

    Transposed<SubMat<'b, T>>
        => Product<Row<'a, T>, Transposed<SubMat<'b, T>>>,

    &'b Transposed<SubMatMut<'c, T>>
        => Product<Row<'a, T>, Transposed<SubMat<'b, T>>>,

    SubMat<'b, T>
        => Product<Row<'a, T>, SubMat<'b, T>>,

    &'b SubMatMut<'c, T>
        => Product<Row<'a, T>, SubMat<'b, T>>,
});

// 4 impls
forward!(Scaled<Product<Row<'a, T>, Chain<'b, T>>> {
    &'b Mat<T>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,

    &'b Transposed<Mat<T>>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,

    &'b Transposed<SubMatMut<'c, T>>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,

    &'b SubMatMut<'c, T>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,
});

// 4 impls
forward!(Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>> {
    &'b Mat<T>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,

    &'b Transposed<Mat<T>>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,

    &'b Transposed<SubMatMut<'c, T>>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,

    &'b SubMatMut<'c, T>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,
});

// 4 impls
forward!(Scaled<Product<Row<'a, T>, SubMat<'b, T>>> {
    &'b Mat<T>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,

    &'b Transposed<Mat<T>>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,

    &'b Transposed<SubMatMut<'c, T>>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,

    &'b SubMatMut<'c, T>
        => Scaled<Product<Row<'a, T>, Chain<'b, T>>>,
});

// 4 impls
forward!(Scaled<Row<'a, T>> {
    &'b Mat<T>
        => Scaled<Product<Row<'a, T>, SubMat<'b, T>>>,

    &'b Transposed<Mat<T>>
        => Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>>,

    &'b Transposed<SubMatMut<'c, T>>
        => Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>>,

    &'b SubMatMut<'c, T>
        => Scaled<Product<Row<'a, T>, SubMat<'b, T>>>,
});
