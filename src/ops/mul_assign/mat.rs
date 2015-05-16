use std::ops::{Mul, MulAssign};

use complex::Complex;

use Forward;
use ops;
use traits::{IterMut, Matrix, MatrixColsMut, MatrixRowsMut, Slice, SliceMut, Transpose};
use {Col, ColMut, Mat, RowMut, Row, Transposed, SubMat, SubMatMut};

// NOTE Core
impl<'a, 'b, A, B> MulAssign<SubMat<'a, A>> for SubMatMut<'b, B> where
    A: Copy,
    B: Copy + Mul<A, Output=B>,
{
    fn mul_assign(&mut self, rhs: SubMat<'a, A>) {
        assert_eq!(self.size(), rhs.size());

        for (lhs, &rhs) in self.iter_mut().zip(rhs) {
            *lhs = *lhs * rhs;
        }
    }
}

// NOTE Core
impl<'a, 'b, A, B> MulAssign<Transposed<SubMat<'a, A>>> for SubMatMut<'b, B> where
    A: Copy,
    B: Copy + Mul<A, Output=B>,
{
    fn mul_assign(&mut self, rhs: Transposed<SubMat<'a, A>>) {
        assert_eq!(self.size(), rhs.size());

        for (mut lhs, rhs) in self.rows_mut().zip(rhs.rows()) {
            for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
                *lhs = *lhs * rhs;
            }
        }
    }
}

// NOTE Secondary
impl<'a, 'b, A, B> MulAssign<Transposed<SubMat<'a, A>>> for Transposed<SubMatMut<'b, B>> where
    A: Copy,
    B: Copy + Mul<A, Output=B>,
{
    fn mul_assign(&mut self, rhs: Transposed<SubMat<'a, A>>) {
        self.0 *= rhs.0
    }
}

// NOTE Secondary
impl<'a, 'b, A, B> MulAssign<SubMat<'a, A>> for Transposed<SubMatMut<'b, B>> where
    A: Copy,
    B: Copy + Mul<A, Output=B>,
{
    fn mul_assign(&mut self, rhs: SubMat<'a, A>) {
        self.0 *= rhs.t()
    }
}

macro_rules! forward {
    ($lhs:ty { $($rhs:ty),+, }) => {
        $(
            impl<'a, 'b, 'c, A, B> MulAssign<$rhs> for $lhs where
                A: Copy + Mul<B, Output=A>, B: Copy,
            {
                fn mul_assign(&mut self, rhs: $rhs) {
                    self.slice_mut(..) *= rhs.slice(..)
                }
            }
         )+
    }
}

forward!(Mat<A> {
    &'a SubMatMut<'b, B>,
    &'a Transposed<SubMatMut<'b, B>>,
    SubMat<'a, B>,
    Transposed<SubMat<'b, B>>,
});

forward!(SubMatMut<'a, A> {
    &'b SubMatMut<'c, B>,
    &'b Transposed<SubMatMut<'c, B>>,
});

forward!(Transposed<Mat<A>> {
    &'a SubMatMut<'b, B>,
    &'a Transposed<SubMatMut<'b, B>>,
    SubMat<'a, B>,
    Transposed<SubMat<'b, B>>,
});

forward!(Transposed<SubMatMut<'a, A>> {
    &'b SubMatMut<'c, B>,
    &'b Transposed<SubMatMut<'c, B>>,
});

macro_rules! scale {
    ($(($lhs:ty, $rhs:ty)),+,) => {
        $(
            // NOTE Core
            impl<'a> MulAssign<$rhs> for SubMatMut<'a, $lhs> {
                fn mul_assign(&mut self, alpha: $rhs) {
                    unsafe {
                        let ref alpha = alpha;

                        if let Some(x) = self.as_slice_mut() {
                            return ops::scal_slice(alpha, x);
                        }

                        if self.0.nrows < self.0.ncols {
                            for RowMut(Row(ref mut x)) in self.rows_mut() {
                                ops::scal_strided(alpha, x);
                            }
                        } else {
                            for ColMut(Col(ref mut x)) in self.cols_mut() {
                                ops::scal_strided(alpha, x);
                            }
                        }
                    }
                }
            }

            // NOTE Secondary
            impl<'a> MulAssign<$rhs> for Transposed<SubMatMut<'a, $lhs>> {
                fn mul_assign(&mut self, alpha: $rhs) {
                    self.0 *= alpha
                }
            }

            // NOTE Forward
            impl MulAssign<$rhs> for Transposed<Mat<$lhs>> {
                fn mul_assign(&mut self, alpha: $rhs) {
                    self.slice_mut(..) *= alpha
                }
            }

            // NOTE Forward
            impl MulAssign<$rhs> for Mat<$lhs> {
                fn mul_assign(&mut self, alpha: $rhs) {
                    self.slice_mut(..) *= alpha
                }
            }
         )+
    }
}

scale! {
    (Complex<f32>, Complex<f32>),
    (Complex<f32>, f32),
    (Complex<f64>, Complex<f64>),
    (Complex<f64>, f64),
    (f32, f32),
    (f64, f64),
}
