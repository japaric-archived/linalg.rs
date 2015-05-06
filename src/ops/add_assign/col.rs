use assign::AddAssign;
use blas::{Axpy, Gemm, Gemv, Transpose};
use onezero::{One, Zero};

use Forward;
use ops::Reduce;
use ops;
use traits::{Matrix, Slice, SliceMut};
use {Chain, Col, ColMut, ColVec, Product, Scaled, Transposed, SubMat};

// Combinations:
//
// LHS: ColMut, ColVec
// RHS: &T, T, Col, &ColMut, &ColVec, Product<Chain, Col>, Product<Transposed<SubMat>, Col>,
//      Product<SubMat, Col>, Scaled<Col>, Scaled<Product<Chain, Col>>,
//      Scaled<Product<Transposed<SubMat>, Col>>, Scaled<Product<SubMat, Col>>
//
// -> 24 implementations

// 5 impls
// Core implementations
impl<'a, 'b, T> AddAssign<&'a T> for ColMut<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        let ColMut(Col(ref mut y)) = *self;
        let x = rhs;
        let ref alpha = T::one();

        ops::axpy_strided_scalar(alpha, x, y)
    }
}

impl<'a, 'b, 'c, T> AddAssign<Scaled<Product<Chain<'a, T>, Col<'b, T>>>> for ColMut<'c, T> where
    T: Gemm + Gemv + One + Zero,
{
    fn add_assign(&mut self, rhs: Scaled<Product<Chain<T>, Col<T>>>) {
        unsafe {
            use ops::reduce::MatMulCol::*;

            assert_eq_size!(self, rhs);

            let Scaled(alpha, rhs) = rhs;
            let ref alpha = alpha;

            let a_mul_b = rhs.reduce();

            let (ref transa, a, x) = match a_mul_b {
                M_C(ref lhs, x) => {
                    (Transpose::No, lhs.slice(..), x)
                },
                M_CV(ref lhs, ref rhs) => {
                    (Transpose::No, lhs.slice(..), rhs.slice(..))
                },
                SM_CV((transa, a), ref rhs) => {
                    (transa, a, rhs.slice(..))
                }
            };

            let y = self.slice_mut(..);
            let ref beta = T::one();

            ops::gemv(transa, alpha, a, beta, x, y)
        }
    }
}

impl<'a, 'b, 'c, T>
AddAssign<Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>> for ColMut<'c, T> where
    T: Gemv + One,
{
    fn add_assign(&mut self, rhs: Scaled<Product<Transposed<SubMat<T>>, Col<T>>>) {
        unsafe {
            assert_eq_size!(self, rhs);

            let ref trans = Transpose::Yes;
            let Scaled(ref alpha, Product(Transposed(a), x)) = rhs;
            let ref beta = T::one();
            let y = ColMut(self.0);

            ops::gemv(trans, alpha, a, beta, x, y);
        }
    }
}

impl<'a, 'b, 'c, T> AddAssign<Scaled<Product<SubMat<'a, T>, Col<'b, T>>>> for ColMut<'c, T> where
    T: Gemv + One,
{
    fn add_assign(&mut self, rhs: Scaled<Product<SubMat<T>, Col<T>>>) {
        unsafe {
            assert_eq_size!(self, rhs);

            let ref trans = Transpose::No;
            let Scaled(ref alpha, Product(a, x)) = rhs;
            let ref beta = T::one();
            let y = ColMut(self.0);

            ops::gemv(trans, alpha, a, beta, x, y);
        }
    }
}

impl<'a, 'b, T> AddAssign<Scaled<Col<'a, T>>> for ColMut<'b, T> where T: Axpy {
    fn add_assign(&mut self, rhs: Scaled<Col<T>>) {
        unsafe {
            assert_eq_size!(self, rhs);

            ops::axpy_strided_strided(&rhs.0, &(rhs.1).0, &mut (self.0).0)
        }
    }
}

// 4 impls
// Secondary implementations
impl<'a, 'b, T> AddAssign<Col<'a, T>> for ColMut<'b, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: Col<T>) {
        self.add_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, 'c, T> AddAssign<Product<Chain<'a, T>, Col<'b, T>>> for ColMut<'c, T> where
    T: Gemm + Gemv + One + Zero,
{
    fn add_assign(&mut self, rhs: Product<Chain<T>, Col<T>>) {
        self.add_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, 'c, T>
AddAssign<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> for ColMut<'c, T> where
    T: Gemv + One,
{
    fn add_assign(&mut self, rhs: Product<Transposed<SubMat<T>>, Col<T>>) {
        self.add_assign(Scaled(T::one(), rhs))
    }
}

impl<'a, 'b, 'c, T> AddAssign<Product<SubMat<'a, T>, Col<'b, T>>> for ColMut<'c, T> where
    T: Gemv + One,
{
    fn add_assign(&mut self, rhs: Product<SubMat<T>, Col<T>>) {
        self.add_assign(Scaled(T::one(), rhs))
    }
}

macro_rules! forward {
    ($lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, 'c, T> AddAssign<$rhs> for $lhs where $(T: $bound),+ {
                fn add_assign(&mut self, rhs: $rhs) {
                    self.slice_mut(..).add_assign(rhs.slice(..))
                }
            }
         )+
    }
}

// 3 impls
forward!(ColMut<'a, T> {
    &'b ColMut<'c, T> { Axpy, One },
    &'b ColVec<T> { Axpy, One },
});

impl<'a, T> AddAssign<T> for ColMut<'a, T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        self.add_assign(&rhs)
    }
}

// 12 impls
forward!(ColVec<T> {
    Col<'a, T> { Axpy, One },
    &'a ColMut<'b, T> { Axpy, One },
    &'a ColVec<T> { Axpy, One },
    Product<Chain<'a, T>, Col<'b, T>> { Gemm, Gemv, One, Zero },
    Product<Transposed<SubMat<'a, T>>, Col<'b, T>> { Gemv, One },
    Product<SubMat<'a, T>, Col<'b, T>> { Gemv, One },
    Scaled<Col<'a, T>> { Axpy },
    Scaled<Product<Chain<'a, T>, Col<'b, T>>> { Gemm, Gemv, One, Zero },
    Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> { Gemv, One },
    Scaled<Product<SubMat<'a, T>, Col<'b, T>>> { Gemv, One },
});

impl<'a, T> AddAssign<&'a T> for ColVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: &T) {
        self.slice_mut(..).add_assign(rhs)
    }
}

impl<T> AddAssign<T> for ColVec<T> where T: Axpy + One {
    fn add_assign(&mut self, rhs: T) {
        self.slice_mut(..).add_assign(&rhs)
    }
}
