use std::ops::Mul;

use Forward;
use traits::{Matrix, Slice};
use {Chain, Col, ColMut, ColVec, Mat, Product, Scaled, Transposed, SubMat, SubMatMut};

// Combinations:
//
// LHS: Chain, &Mat, Scaled<Chain>, Scaled<Transposed<SubMat>>, Scaled<SubMat>, &Transposed<Mat>,
//      Transposed<SubMat>, &Transposed<SubMatMut>, SubMat, &SubMatMut
// RHS: Col, &ColMut, &ColVec, Product<Chain, Col>, Product<Transposed<SubMat>, Col>,
//      Product<SubMat, Col>, Scaled<Col>, Scaled<Product<Chain, Col>>,
//      Scaled<Product<Transposed<SubMat>, Col>>, Scaled<Product<SubMat, Col>>
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
mul!(Chain<'a, T>, Col<'b, T>);
mul!(Transposed<SubMat<'a, T>>, Col<'b, T>);
mul!(SubMat<'a, T>, Col<'b, T>);

macro_rules! product {
    ($lhs:ty { $($rhs:ty),+ }) => {
        $(
            // Secondary implementations
            impl<'a, 'b, T> Mul<Product<$rhs, Col<'b, T>>> for $lhs {
                type Output = Product<Chain<'a, T>, Col<'b, T>>;

                fn mul(self, rhs: Product<$rhs, Col<'b, T>>) -> Product<Chain<'a, T>, Col<'b, T>> {
                    Product(self * rhs.0, rhs.1)
                }
            }

            impl<'a, 'b, T> Mul<Scaled<Product<$rhs, Col<'b, T>>>> for $lhs {
                type Output = Scaled<Product<Chain<'a, T>, Col<'b, T>>>;

                fn mul(
                    self,
                    rhs: Scaled<Product<$rhs, Col<'b, T>>>,
                ) -> Scaled<Product<Chain<'a, T>, Col<'b, T>>> {
                    Scaled(rhs.0, self * rhs.1)
                }
            }

            impl<'a, 'b, T> Mul<Product<$rhs, Col<'b, T>>> for Scaled<$lhs> {
                type Output = Scaled<Product<Chain<'a, T>, Col<'b, T>>>;

                fn mul(
                    self,
                    rhs: Product<$rhs, Col<'b, T>>,
                ) -> Scaled<Product<Chain<'a, T>, Col<'b, T>>> {
                    Scaled(self.0, self.1 * rhs)
                }
            }

            impl<'a, 'b, T> Mul<Scaled<Product<$rhs, Col<'b, T>>>> for Scaled<$lhs> where
                T: Mul<Output=T>,
            {
                type Output = Scaled<Product<Chain<'a, T>, Col<'b, T>>>;

                fn mul(
                    self,
                    rhs: Scaled<Product<$rhs, Col<'b, T>>>,
                ) -> Scaled<Product<Chain<'a, T>, Col<'b, T>>> {
                    Scaled(self.0 * rhs.0, self.1 * rhs.1)
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

// 2 impls
forward!(Chain<'a, T> {
    &'b ColMut<'c, T>
        => Product<Chain<'a, T>, Col<'b, T>>,

    &'b ColVec<T>
        => Product<Chain<'a, T>, Col<'b, T>>,
});

// 10 impls
forward!(&'a Mat<T> {
    Col<'b, T>
        => Product<SubMat<'a, T>, Col<'b, T>>,

    &'b ColMut<'c, T>
        => Product<SubMat<'a, T>, Col<'b, T>>,

    &'b ColVec<T>
        => Product<SubMat<'a, T>, Col<'b, T>>,

    Product<Chain<'a, T>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Product<Transposed<SubMat<'a, T>>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Product<SubMat<'a, T>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Scaled<Col<'b, T>>
        => Scaled<Product<SubMat<'a, T>, Col<'b, T>>>,

    Scaled<Product<Chain<'a, T>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,

    Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,

    Scaled<Product<SubMat<'a, T>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,
});

// 2 impls
forward!(Scaled<Chain<'a, T>> {
    &'b ColMut<'c, T>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,

    &'b ColVec<T>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,
});

// 2 impls
forward!(Scaled<Transposed<SubMat<'a, T>>> {
    &'b ColMut<'c, T>
        => Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>,

    &'b ColVec<T>
        => Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>,
});

// 2 impls
forward!(Scaled<SubMat<'a, T>> {
    &'b ColMut<'c, T>
        => Scaled<Product<SubMat<'a, T>, Col<'b, T>>>,

    &'b ColVec<T>
        => Scaled<Product<SubMat<'a, T>, Col<'b, T>>>,
});

// 10 impls
forward!(&'a Transposed<Mat<T>> {
    Col<'b, T>
        => Product<Transposed<SubMat<'a, T>>, Col<'b, T>>,

    &'b ColMut<'c, T>
        => Product<Transposed<SubMat<'a, T>>, Col<'b, T>>,

    &'b ColVec<T>
        => Product<Transposed<SubMat<'a, T>>, Col<'b, T>>,

    Product<Chain<'a, T>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Product<Transposed<SubMat<'a, T>>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Product<SubMat<'a, T>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Scaled<Col<'b, T>>
        => Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>,

    Scaled<Product<Chain<'a, T>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,

    Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,

    Scaled<Product<SubMat<'a, T>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,
});

// 2 impls
forward!(Transposed<SubMat<'a, T>> {
    &'b ColMut<'c, T>
        => Product<Transposed<SubMat<'a, T>>, Col<'b, T>>,

    &'b ColVec<T>
        => Product<Transposed<SubMat<'a, T>>, Col<'b, T>>,
});

// 10 impls
forward!(&'a Transposed<SubMatMut<'b, T>> {
    Col<'c, T>
        => Product<Transposed<SubMat<'a, T>>, Col<'c, T>>,

    &'c ColMut<'d, T>
        => Product<Transposed<SubMat<'a, T>>, Col<'c, T>>,

    &'c ColVec<T>
        => Product<Transposed<SubMat<'a, T>>, Col<'c, T>>,

    Product<Chain<'a, T>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Product<Transposed<SubMat<'a, T>>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Product<SubMat<'a, T>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Scaled<Col<'b, T>>
        => Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>,

    Scaled<Product<Chain<'a, T>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,

    Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,

    Scaled<Product<SubMat<'a, T>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,
});

// 2 impls
forward!(SubMat<'a, T> {
    &'b ColMut<'c, T>
        => Product<SubMat<'a, T>, Col<'b, T>>,

    &'b ColVec<T>
        => Product<SubMat<'a, T>, Col<'b, T>>,
});

// 10 impls
forward!(&'a SubMatMut<'b, T> {
    Col<'c, T>
        => Product<SubMat<'a, T>, Col<'c, T>>,

    &'c ColMut<'d, T>
        => Product<SubMat<'a, T>, Col<'c, T>>,

    &'c ColVec<T>
        => Product<SubMat<'a, T>, Col<'c, T>>,

    Product<Chain<'a, T>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Product<Transposed<SubMat<'a, T>>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Product<SubMat<'a, T>, Col<'b, T>>
        => Product<Chain<'a, T>, Col<'b, T>>,

    Scaled<Col<'b, T>>
        => Scaled<Product<SubMat<'a, T>, Col<'b, T>>>,

    Scaled<Product<Chain<'a, T>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,

    Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,

    Scaled<Product<SubMat<'a, T>, Col<'b, T>>>
        => Scaled<Product<Chain<'a, T>, Col<'b, T>>>,
});
