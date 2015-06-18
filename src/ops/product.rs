#![doc(hidden)]

use std::ops::Mul;

use ops::Product;
use order::Order;

macro_rules! mat_col {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T, O> Mul<$rhs> for $lhs where O: Order {
            type Output = Product<&'a ::ops::Mat<T>, &'b ::strided::Col<T>>;

            fn mul(self, rhs: $rhs) -> Self::Output {
                Product(self, rhs)
            }
        }
    }
}

mat_col!(&'a ::Mat<T, O>, &'b ::Col<T>);
mat_col!(&'a ::Mat<T, O>, &'b ::strided::Col<T>);
mat_col!(&'a ::strided::Mat<T, O>, &'b ::Col<T>);
mat_col!(&'a ::strided::Mat<T, O>, &'b ::strided::Col<T>);

macro_rules! mat_mat {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T, O1, O2> Mul<$rhs> for $lhs where O1: Order, O2: Order {
            type Output = Product<&'a ::ops::Mat<T>, &'b ::ops::Mat<T>>;

            fn mul(self, rhs: $rhs) -> Self::Output {
                Product(self, rhs)
            }
        }
    }
}

mat_mat!(&'a ::Mat<T, O1>, &'b ::Mat<T, O2>);
mat_mat!(&'a ::Mat<T, O1>, &'b ::strided::Mat<T, O2>);
mat_mat!(&'a ::strided::Mat<T, O1>, &'b ::Mat<T, O2>);
mat_mat!(&'a ::strided::Mat<T, O1>, &'b ::strided::Mat<T, O2>);

macro_rules! row_mat {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T, O> Mul<$rhs> for $lhs where O: Order {
            type Output = Product<&'a ::strided::Row<T>, &'b ::ops::Mat<T>>;

            fn mul(self, rhs: $rhs) -> Self::Output {
                Product(self, rhs)
            }
        }
    }
}

row_mat!(&'a ::Row<T>, &'b ::Mat<T, O>);
row_mat!(&'a ::Row<T>, &'b ::strided::Mat<T, O>);
row_mat!(&'a ::strided::Row<T>, &'b ::Mat<T, O>);
row_mat!(&'a ::strided::Row<T>, &'b ::strided::Mat<T, O>);
