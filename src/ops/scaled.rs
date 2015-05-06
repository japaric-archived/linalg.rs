#![doc(hidden)]

use std::ops::{Div, Mul, Neg};

use complex::Complex;
use onezero::One;

use {
    Chain, Col, ColMut, ColVec, Mat, Product, Row, RowMut, RowVec, Scaled, Transposed, SubMat,
    SubMatMut,
};
use traits::{Matrix, Slice};

// NOTE Core
impl<T, M> Div<T> for Scaled<M> where M: Matrix<Elem=T>, T: Div<Output=T> {
    type Output = Scaled<M>;

    fn div(self, rhs: T) -> Scaled<M> {
        Scaled(self.0 / rhs, self.1)
    }
}

// NOTE Core
impl<T, L, R> Div<T> for Product<L, R> where
    L: Matrix<Elem=T>,
    R: Matrix<Elem=T>,
    T: Div<Output=T> + One,
{
    type Output = Scaled<Product<L, R>>;

    fn div(self, rhs: T) -> Scaled<Product<L, R>> {
        Scaled(T::one() / rhs, self)
    }
}

// NOTE Core
impl<T, M> Neg for Scaled<M> where M: Matrix<Elem=T>, T: Neg<Output=T> {
    type Output = Scaled<M>;

    fn neg(self) -> Scaled<M> {
        Scaled(self.0.neg(), self.1)
    }
}

// NOTE Core
impl<'a, T> Mul<T> for Chain<'a, T> {
    type Output = Scaled<Chain<'a, T>>;

    fn mul(self, rhs: T) -> Scaled<Chain<'a, T>> {
        Scaled(rhs, self)
    }
}

macro_rules! chain {
    ($($t:ty),+) => {
        $(
            // NOTE Secondary
            impl<'a> Mul<Chain<'a, $t>> for $t {
                type Output = Scaled<Chain<'a, $t>>;

                fn mul(self, rhs: Chain<'a, $t>) -> Scaled<Chain<'a, $t>> {
                    rhs * self
                }
            }
         )+
    };
}

chain!(f32, f64, Complex<f32>, Complex<f64>);

macro_rules! mul {
    ($ty:ident $ty_mut:ident $ty_owned:ident for $($t:ty),+) => {
        // Core implementations
        impl<'a, T> Mul<T> for $ty<'a, T> {
            type Output = Scaled<$ty<'a, T>>;

            fn mul(self, rhs: T) -> Scaled<$ty<'a, T>> {
                Scaled(rhs, self)
            }
        }

        // "Forwarding" implementations
        impl<'a, 'b, T> Mul<T> for &'a $ty_mut<'b, T> {
            type Output = Scaled<$ty<'a, T>>;

            fn mul(self, rhs: T) -> Scaled<$ty<'a, T>> {
                self.slice(..) * rhs
            }
        }

        impl<'a, T> Mul<T> for &'a $ty_owned<T> {
            type Output = Scaled<$ty<'a, T>>;

            fn mul(self, rhs: T) -> Scaled<$ty<'a, T>> {
                self.slice(..) * rhs
            }
        }

        impl<T> Mul<T> for $ty_owned<T> {
            type Output = Scaled<$ty_owned<T>>;

            fn mul(self, rhs: T) -> Scaled<$ty_owned<T>> {
                Scaled(rhs, self)
            }
        }

        // Reverse operations
        $(
            impl<'a> Mul<$ty<'a, $t>> for $t {
                type Output = Scaled<$ty<'a, $t>>;

                fn mul(self, rhs: $ty<$t>) -> Scaled<$ty<$t>> {
                    rhs * self
                }
            }

            impl<'a, 'b> Mul<&'a $ty_mut<'b, $t>> for $t {
                type Output = Scaled<$ty<'a, $t>>;

                fn mul(self, rhs: &'a $ty_mut<$t>) -> Scaled<$ty<'a, $t>> {
                    rhs * self
                }
            }

            impl<'a> Mul<&'a $ty_owned<$t>> for $t {
                type Output = Scaled<$ty<'a, $t>>;

                fn mul(self, rhs: &'a $ty_owned<$t>) -> Scaled<$ty<'a, $t>> {
                    rhs * self
                }
            }

            impl Mul<$ty_owned<$t>> for $t {
                type Output = Scaled<$ty_owned<$t>>;

                fn mul(self, rhs: $ty_owned<$t>) -> Scaled<$ty_owned<$t>> {
                    rhs * self
                }
            }
         )+
    };
}

mul!(Col ColMut ColVec for f32, f64, Complex<f32>, Complex<f64>);
mul!(Row RowMut RowVec for f32, f64, Complex<f32>, Complex<f64>);
mul!(SubMat SubMatMut Mat for f32, f64, Complex<f32>, Complex<f64>);

impl<T, L, R> Mul<T> for Product<L, R> where
    L: Matrix<Elem=T>,
    R: Matrix<Elem=T>,
{
    type Output = Scaled<Product<L, R>>;

    fn mul(self, rhs: T) -> Scaled<Product<L, R>> {
        Scaled(rhs, self)
    }
}

macro_rules! product {
    ($($t:ty),+) => {
        $(
            impl<L, R> Mul<Product<L, R>> for $t where
                L: Matrix<Elem=$t>,
                R: Matrix<Elem=$t>,
            {
                type Output = Scaled<Product<L, R>>;

                fn mul(self, rhs: Product<L, R>) -> Scaled<Product<L, R>> {
                    rhs * self
                }
            }
         )+
    };
}

product!(f32, f64, Complex<f32>, Complex<f64>);

impl<T, M> Mul<T> for Scaled<M> where
    M: Matrix<Elem=T>,
    T: Mul<Output=T>,
{
    type Output = Scaled<M>;

    fn mul(self, rhs: T) -> Scaled<M> {
        Scaled(self.0 * rhs, self.1)
    }
}

macro_rules! scaled {
    ($($t:ty),+) => {
        $(
            impl<M> Mul<Scaled<M>> for $t where M: Matrix<Elem=$t> {
                type Output = Scaled<M>;

                fn mul(self, rhs: Scaled<M>) -> Scaled<M> {
                    Scaled(self * rhs.0, rhs.1)
                }
            }
         )+
    };
}

scaled!(f32, f64, Complex<f32>, Complex<f64>);

impl<T> Mul<T> for Transposed<Mat<T>> {
    type Output = Scaled<Transposed<Mat<T>>>;

    fn mul(self, rhs: T) -> Scaled<Transposed<Mat<T>>> {
        Scaled(rhs, self)
    }
}

impl<'a, T> Mul<T> for &'a Transposed<Mat<T>> {
    type Output = Scaled<Transposed<SubMat<'a, T>>>;

    fn mul(self, rhs: T) -> Scaled<Transposed<SubMat<'a, T>>> {
        self.slice(..) * rhs
    }
}

impl<'a, T> Mul<T> for Transposed<SubMat<'a, T>> {
    type Output = Scaled<Transposed<SubMat<'a, T>>>;

    fn mul(self, rhs: T) -> Scaled<Transposed<SubMat<'a, T>>> {
        Scaled(rhs, self)
    }
}

impl<'a, 'b, T> Mul<T> for &'a Transposed<SubMatMut<'b, T>> {
    type Output = Scaled<Transposed<SubMat<'a, T>>>;

    fn mul(self, rhs: T) -> Scaled<Transposed<SubMat<'a, T>>> {
        self.slice(..) * rhs
    }
}

macro_rules! transposed {
    ($($t:ty),+) => {
        $(
            impl Mul<Transposed<Mat<$t>>> for $t {
                type Output = Scaled<Transposed<Mat<$t>>>;

                fn mul(self, rhs: Transposed<Mat<$t>>) -> Scaled<Transposed<Mat<$t>>> {
                    rhs * self
                }
            }

            impl<'a> Mul<&'a Transposed<Mat<$t>>> for $t {
                type Output = Scaled<Transposed<SubMat<'a, $t>>>;

                fn mul(self, rhs: &'a Transposed<Mat<$t>>) -> Scaled<Transposed<SubMat<'a, $t>>> {
                    rhs * self
                }
            }

            impl<'a> Mul<Transposed<SubMat<'a, $t>>> for $t {
                type Output = Scaled<Transposed<SubMat<'a, $t>>>;

                fn mul(
                    self,
                    rhs: Transposed<SubMat<'a, $t>>,
                ) -> Scaled<Transposed<SubMat<'a, $t>>> {
                    rhs * self
                }
            }

            impl<'a, 'b> Mul<&'a Transposed<SubMatMut<'b, $t>>> for $t {
                type Output = Scaled<Transposed<SubMat<'a, $t>>>;

                fn mul(
                    self,
                    rhs: &'a Transposed<SubMatMut<'b, $t>>,
                ) -> Scaled<Transposed<SubMat<'a, $t>>> {
                    rhs * self
                }
            }
         )+
    };
}

transposed!(f32, f64, Complex<f32>, Complex<f64>);
