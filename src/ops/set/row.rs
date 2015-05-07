use blas::{Copy, Gemv};

use Forward;
use onezero::{One, Zero};
use ops::{set, self};
use traits::Transpose as _0;
use traits::{Matrix, Set, Slice, SliceMut};
use {Product, RowMut, Row, RowVec, Scaled, SubMat, Transposed};

// NOTE Core
impl<'a, T> Set<T> for RowMut<'a, T> where T: Copy {
    fn set(&mut self, value: T) {
        let RowMut(Row(ref mut y)) = *self;
        let ref x = value;

        set::strided(x, y)
    }
}

// NOTE Core
impl<'a, 'b, T> Set<Row<'a, T>> for RowMut<'b, T> where T: Copy {
    fn set(&mut self, rhs: Row<T>) {
        unsafe {
            assert_eq!(self.ncols(), rhs.ncols());

            let RowMut(Row(ref mut y)) = *self;
            let Row(ref x) = rhs;

            ops::copy_strided(x, y)
        }
    }
}

// NOTE Secondary
impl<'a, 'b, 'c, T> Set<Scaled<Product<Row<'b, T>, SubMat<'a, T>>>> for RowMut<'c, T> where
    T: Gemv + Zero,
{
    fn set(&mut self, rhs: Scaled<Product<Row<T>, SubMat<T>>>) {
        self.slice_mut(..).t().set(rhs.t())
    }
}

// NOTE Secondary
impl<'a, 'b, 'c, T>
Set<Scaled<Product<Row<'b, T>, Transposed<SubMat<'a, T>>>>> for RowMut<'c, T> where
    T: Gemv + Zero,
{
    fn set(&mut self, rhs: Scaled<Product<Row<T>, Transposed<SubMat<T>>>>) {
        self.slice_mut(..).t().set(rhs.t())
    }
}

// NOTE Secondary
impl<'a, 'b, 'c, T> Set<Product<Row<'b, T>, SubMat<'a, T>>> for RowMut<'c, T> where
    T: Gemv + One + Zero,
{
    fn set(&mut self, rhs: Product<Row<T>, SubMat<T>>) {
        self.set(Scaled(T::one(), rhs))
    }
}

// NOTE Secondary
impl<'a, 'b, 'c, T> Set<Product<Row<'b, T>, Transposed<SubMat<'a, T>>>> for RowMut<'c, T> where
    T: Gemv + One + Zero,
{
    fn set(&mut self, rhs: Product<Row<T>, Transposed<SubMat<T>>>) {
        self.set(Scaled(T::one(), rhs))
    }
}


// NOTE Forward
impl<T> Set<T> for RowVec<T> where T: Copy {
    fn set(&mut self, value: T) {
        self.slice_mut(..).set(value)
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

forward!(RowMut<'a, T> {
    &'b RowMut<'c, T> { Copy },
    &'b RowVec<T> { Copy },
});

forward!(RowVec<T> {
    &'a RowMut<'b, T> { Copy },
    &'a RowVec<T> { Copy },
    Product<Row<'a, T>, SubMat<'b, T>> { Gemv, One, Zero },
    Product<Row<'a, T>, Transposed<SubMat<'b, T>>> { Gemv, One, Zero },
    Scaled<Product<Row<'a, T>, SubMat<'b, T>>> { Gemv, Zero },
    Scaled<Product<Row<'a, T>, Transposed<SubMat<'b, T>>>> { Gemv, Zero },
});
