use blas::{Copy, Gemm, Transpose};

use Forward;
use onezero::{One, Zero};
use ops::{Reduce, set, self};
use traits::Transpose as _0;
use traits::{Matrix, MatrixCols, MatrixColsMut, MatrixRows, MatrixRowsMut, Set, Slice, SliceMut};
use {Chain, Col, ColMut, Mat, Row, RowMut, Transposed, Scaled, SubMat, SubMatMut};

// NOTE Core
impl<'a, T> Set<T> for SubMatMut<'a, T> where T: Copy {
    fn set(&mut self, value: T) {
        let ref x = value;

        if let Some(y) = self.as_slice_mut() {
            return set::slice(x, y);
        }

        if self.nrows() < self.ncols() {
            for RowMut(Row(ref mut y)) in self.rows_mut() {
                set::strided(x, y)
            }
        } else {
            for ColMut(Col(ref mut y)) in self.cols_mut() {
                set::strided(x, y)
            }
        }
    }
}

// NOTE Core
impl<'a, 'b, T> Set<SubMat<'a, T>> for SubMatMut<'b, T> where T: Copy {
    fn set(&mut self, rhs: SubMat<T>) {
        unsafe {
            assert_eq!(self.size(), rhs.size());

            if let (Some(y), Some(x)) = (self.as_slice_mut(), rhs.as_slice()) {
                return ops::copy_slice(x, y);
            }

            if self.nrows() < self.ncols() {
                for (RowMut(Row(ref mut y)), Row(ref x)) in self.rows_mut().zip(rhs.rows()) {
                    ops::copy_strided(x, y)
                }
            } else {
                for (ColMut(Col(ref mut y)), Col(ref x)) in self.cols_mut().zip(rhs.cols()) {
                    ops::copy_strided(x, y)
                }
            }
        }
    }
}

// NOTE Core
impl<'a, 'b, T> Set<Transposed<SubMat<'a, T>>> for SubMatMut<'b, T> where T: Copy {
    fn set(&mut self, rhs: Transposed<SubMat<T>>) {
        unsafe {
            assert_eq!(self.size(), rhs.size());

            if self.nrows() < self.ncols() {
                for (RowMut(Row(ref mut y)), Row(ref x)) in self.rows_mut().zip(rhs.rows()) {
                    ops::copy_strided(x, y)
                }
            } else {
                for (ColMut(Col(ref mut y)), Col(ref x)) in self.cols_mut().zip(rhs.cols()) {
                    ops::copy_strided(x, y)
                }
            }
        }
    }
}

// NOTE Core
impl<'a, 'b, T> Set<Scaled<Chain<'a, T>>> for SubMatMut<'b, T> where T: Gemm + One + Zero {
    fn set(&mut self, rhs: Scaled<Chain<T>>) {
        unsafe {
            use ops::reduce::MatMulMat::*;

            assert_eq!(self.size(), rhs.size());

            let Scaled(alpha, rhs) = rhs;
            let ref alpha = alpha;
            let a_mul_b = rhs.reduce();

            let ((ref transa, a), (ref transb, b)) = match a_mul_b {
                M_M(ref lhs, ref rhs) => {
                    ((Transpose::No, lhs.slice(..)), (Transpose::No, rhs.slice(..)))
                },
                M_SM(ref lhs, rhs) => ((Transpose::No, lhs.slice(..)), rhs),
                SM_M(lhs, ref rhs) => (lhs, (Transpose::No, rhs.slice(..))),
                SM_SM(lhs, rhs) => (lhs, rhs),
            };

            let c = self.slice_mut(..);
            let ref beta = T::zero();

            ops::gemm(transa, transb, alpha, a, b, beta, c)
        }
    }
}

// NOTE Secondary
impl<'a, T> Set<T> for Transposed<SubMatMut<'a, T>> where T: Copy {
    fn set(&mut self, value: T) {
        self.0.set(value)
    }
}

// NOTE Secondary
impl<'a, 'b, T> Set<Chain<'a, T>> for SubMatMut<'b, T> where T: Gemm + One + Zero {
    fn set(&mut self, rhs: Chain<T>) {
        self.set(Scaled(T::one(), rhs))
    }
}

// NOTE Secondary
impl<'a, 'b, T> Set<Chain<'a, T>> for Transposed<SubMatMut<'b, T>> where T: Gemm + One + Zero {
    fn set(&mut self, rhs: Chain<T>) {
        self.set(Scaled(T::one(), rhs))
    }
}

// NOTE Secondary
impl<'a, 'b, T> Set<Scaled<Chain<'a, T>>> for Transposed<SubMatMut<'b, T>> where
    T: Gemm + One + Zero,
{
    fn set(&mut self, rhs: Scaled<Chain<T>>) {
        self.0.set(rhs.t())
    }
}

// NOTE Secondary
impl<'a, 'b, T> Set<SubMat<'a, T>> for Transposed<SubMatMut<'b, T>> where T: Copy {
    fn set(&mut self, rhs: SubMat<T>) {
        self.0.set(Transposed(rhs))
    }
}

// NOTE Secondary
impl<'a, 'b, T> Set<Transposed<SubMat<'a, T>>> for Transposed<SubMatMut<'b, T>> where T: Copy {
    fn set(&mut self, rhs: Transposed<SubMat<T>>) {
        self.0.set(rhs.0)
    }
}

// NOTE Forward
impl<T> Set<T> for Transposed<Mat<T>> where T: Copy {
    fn set(&mut self, value: T) {
        self.slice_mut(..).set(value)
    }
}

// NOTE Forward
impl<T> Set<T> for Mat<T> where T: Copy {
    fn set(&mut self, value: T) {
        self.slice_mut(..).set(value)
    }
}

macro_rules! forward {
    ($lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, 'c, T> Set<$rhs> for $lhs where $(T: $bound),+ {
                fn set(&mut self, rhs: $rhs) {
                    self.slice_mut(..).set(rhs.slice(..))
                }
            }
         )+
    }
}

forward!(Mat<T> {
    Chain<'a, T> { Gemm, One, Zero },
    Scaled<Chain<'a, T>> { Gemm, One, Zero },
    SubMat<'a, T> { Copy },
    &'a SubMatMut<'b, T> { Copy },
});

forward!(Transposed<Mat<T>> {
    Scaled<Chain<'a, T>> { Gemm, One, Zero },
    Chain<'a, T> { Gemm, One, Zero },
    SubMat<'a, T> { Copy },
    &'a SubMatMut<'b, T> { Copy },
});

forward!(Transposed<SubMatMut<'a, T>> {
    &'b SubMatMut<'c, T> { Copy },
    &'b Transposed<SubMatMut<'c, T>> { Copy },
});

forward!(SubMatMut<'a, T> {
    &'b SubMatMut<'c, T> { Copy },
    &'b Transposed<SubMatMut<'c, T>> { Copy },
});
