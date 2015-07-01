//! Matrix operations

mod add;
mod add_assign;
mod chain;
mod copy;
mod div_assign;
mod mat;
mod mul;
mod mul_assign;
mod neg;
mod norm;
mod product;
mod scaled;
mod sub;
mod sub_assign;
mod sum;

pub mod blas;

use std::fat_ptr;
use std::raw::FatPtr;

/// Lazy matrix multiplication
pub struct Chain<'a, T: 'a> {
    first: &'a Mat<T>,
    second: &'a Mat<T>,
    tail: Vec<&'a Mat<T>>,
}

/// A matrix whose `Order` has been "erased" from its type signature
///
/// NOTE You almost never want to directly use this type, this type is a byproduct of arithmetic
/// operations
pub unsized type Mat<T>;

/// Lazy multiplication
// NB Combinations:
//
// - Col-like: `Product<&Mat, &Col>`
// - Mat-like: `Product<&Mat, &Mat>`
// - Row-like: `Product<&Row, &Mat>`
pub struct Product<L, R>(L, R);

/// Lazily scaled matrix
// NB Possible values of `M`: `Chain`, `&[mut] Col`, `&[mut] Mat`, `Product`, `&[mut] Row`
pub struct Scaled<M>(M::Elem, M) where M: Matrix;

/// Lazy addition
pub struct Sum<L, R>(L, R);

impl<T> Mat<T> {
    fn repr(&self) -> FatPtr<T, ::ops::mat::Info> {
        fat_ptr::repr(self)
    }
}

impl ::Order {
    fn trans(&self) -> ::blas::Transpose {
        match *self {
            ::Order::Col => ::blas::Transpose::No,
            ::Order::Row => ::blas::Transpose::Yes,
        }
    }
}

// NB These "impls" live here so they show up in the docs -- the contents of other modules are
// ![doc(hidden)]
use traits::{Matrix, Transpose};

impl<T, L, R> Matrix for Product<L, R> where L: Matrix<Elem=T>, R: Matrix<Elem=T> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        self.0.nrows()
    }

    fn ncols(&self) -> u32 {
        self.1.ncols()
    }
}

impl<L, R> Transpose for Product<L, R> where L: Transpose, R: Transpose {
    type Output = Product<R::Output, L::Output>;

    fn t(self) -> Product<R::Output, L::Output> {
        Product(self.1.t(), self.0.t())
    }
}

impl<M> Matrix for Scaled<M> where M: Matrix {
    type Elem = M::Elem;

    fn nrows(&self) -> u32 {
        self.1.nrows()
    }

    fn ncols(&self) -> u32 {
        self.1.ncols()
    }

    fn size(&self) -> (u32, u32) {
        self.1.size()
    }
}

impl<T, M> Transpose for Scaled<M> where
    M: Matrix<Elem=T> + Transpose,
    M::Output: Matrix<Elem=T>,
{
    type Output = Scaled<M::Output>;

    fn t(self) -> Scaled<M::Output> {
        Scaled(self.0, self.1.t())
    }
}
