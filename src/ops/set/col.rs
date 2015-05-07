use blas::{Copy, Gemv, Transpose};

use Forward;
use onezero::{One, Zero};
use ops::{set, self};
use traits::{Matrix, Set, Slice, SliceMut};
use {ColMut, Col, ColVec, Product, Scaled, Transposed, SubMat};

// NOTE Core
impl<'a, T> Set<T> for ColMut<'a, T> where T: Copy {
    fn set(&mut self, value: T) {
        let ColMut(Col(ref mut y)) = *self;
        let ref x = value;

        set::strided(x, y)
    }
}

// NOTE Core
impl<'a, 'b, T> Set<Col<'a, T>> for ColMut<'b, T> where T: Copy {
    fn set(&mut self, rhs: Col<T>) {
        unsafe {
            assert_eq!(self.nrows(), rhs.nrows());

            let ColMut(Col(ref mut y)) = *self;
            let Col(ref x) = rhs;

            ops::copy_strided(x, y)
        }
    }
}

// NOTE Core
impl<'a, 'b, 'c, T> Set<Scaled<Product<SubMat<'a, T>, Col<'b, T>>>> for ColMut<'c, T> where
    T: Gemv + Zero,
{
    fn set(&mut self, rhs: Scaled<Product<SubMat<T>, Col<T>>>) {
        unsafe {
            assert_eq!(self.nrows(), rhs.nrows());

            let Scaled(ref alpha, Product(a, x)) = rhs;
            let ref trans = Transpose::No;
            let y = self.slice_mut(..);
            let ref beta = T::zero();

            ops::gemv(trans, alpha, a, beta, x, y)
        }
    }
}

// NOTE Core
impl<'a, 'b, 'c, T>
Set<Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>>> for ColMut<'c, T> where
    T: Gemv + Zero,
{
    fn set(&mut self, rhs: Scaled<Product<Transposed<SubMat<T>>, Col<T>>>) {
        unsafe {
            assert_eq!(self.nrows(), rhs.nrows());

            let Scaled(ref alpha, Product(Transposed(a), x)) = rhs;
            let ref trans = Transpose::Yes;
            let y = self.slice_mut(..);
            let ref beta = T::zero();

            ops::gemv(trans, alpha, a, beta, x, y)
        }
    }
}

// NOTE Secondary
impl<'a, 'b, 'c, T> Set<Product<SubMat<'a, T>, Col<'b, T>>> for ColMut<'c, T> where
    T: Gemv + One + Zero,
{
    fn set(&mut self, rhs: Product<SubMat<T>, Col<T>>) {
        self.set(Scaled(T::one(), rhs))
    }
}

// NOTE Secondary
impl<'a, 'b, 'c, T> Set<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> for ColMut<'c, T> where
    T: Gemv + One + Zero,
{
    fn set(&mut self, rhs: Product<Transposed<SubMat<T>>, Col<T>>) {
        self.set(Scaled(T::one(), rhs))
    }
}

// NOTE Forward
impl<T> Set<T> for ColVec<T> where T: Copy {
    fn set(&mut self, value: T) {
        self.slice_mut(..).set(value)
    }
}

// NOTE Forward
impl<'a, T> Set<Col<'a, T>> for ColVec<T> where T: Copy {
    fn set(&mut self, rhs: Col<T>) {
        self.slice_mut(..).set(rhs)
    }
}

macro_rules! forward {
    ($lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            // NOTE Forward
            impl<'a, 'b, 'c, T> Set<$rhs> for $lhs where $(T: $bound),+ {
                fn set(&mut self, rhs: $rhs) {
                    self.slice_mut(..).set(rhs.slice(..))
                }
            }
         )+
    }
}

forward!(ColMut<'a, T> {
    &'b ColMut<'c, T> { Copy },
    &'b ColVec<T> { Copy },
});

forward!(ColVec<T> {
    &'a ColMut<'b, T> { Copy },
    &'a ColVec<T> { Copy },
    Product<SubMat<'a, T>, Col<'b, T>> { Gemv, One, Zero },
    Product<Transposed<SubMat<'a, T>>, Col<'b, T>> { Gemv, One, Zero },
    Scaled<Product<SubMat<'a, T>, Col<'b, T>>> { Gemv, Zero },
    Scaled<Product<Transposed<SubMat<'a, T>>, Col<'b, T>>> { Gemv, Zero },
});
