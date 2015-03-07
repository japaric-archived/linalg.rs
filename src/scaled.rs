use complex::Complex;
use std::ops::Mul;

use {Col, ColVec, Cols, Mat, MutCol, MutRow, MutView, Row, RowVec, Rows, Scaled, Trans, View};
use traits::{Matrix, MatrixCol, MatrixRow};

macro_rules! mul1 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a> Mul<$rhs> for $lhs {
            type Output = Scaled<$lhs, $rhs>;

            fn mul(self, rhs: $rhs) -> Scaled<$lhs, $rhs> {
                rhs * self
            }
        }
    };
    ($lhs:ty, $rhs:ty, $output:ty) => {
        impl<'a> Mul<$rhs> for $lhs {
            type Output = Scaled<$lhs, $output>;

            fn mul(self, rhs: $rhs) -> Scaled<$lhs, $output> {
                rhs * self
            }
        }
    };
}

macro_rules! mul2 {
    ($lhs:ty, $rhs:ty, $output:ty) => {
        impl<'a, 'b> Mul<$rhs> for $lhs {
            type Output = Scaled<$lhs, $output>;

            fn mul(self, rhs: $rhs) -> Scaled<$lhs, $output> {
                rhs * self
            }
        }
    };
}

impl<'a, T, M> Iterator for Scaled<T, Rows<'a, M>> where
    T: 'a + Clone,
    M: MatrixRow + Matrix<Elem=T>,
{
    type Item = Scaled<T, Row<'a, T>>;

    fn next(&mut self) -> Option<Scaled<T, Row<'a, T>>> {
        self.1.next().map(|r| Scaled(self.0.clone(), r))
    }
}

impl<'a, T, M> Iterator for Scaled<T, Cols<'a, M>> where
    T: 'a + Clone,
    M: MatrixCol + Matrix<Elem=T>,
{
    type Item = Scaled<T, Col<'a, T>>;

    fn next(&mut self) -> Option<Scaled<T, Col<'a, T>>> {
        self.1.next().map(|r| Scaled(self.0.clone(), r))
    }
}

impl<T, M> Matrix for Scaled<T, M> where M: Matrix<Elem=T> {
    type Elem = T;

    fn ncols(&self) -> usize {
        self.1.ncols()
    }

    fn nrows(&self) -> usize {
        self.1.nrows()
    }

    fn size(&self) -> (usize, usize) {
        self.1.size()
    }
}

// col
impl<'a, T> Mul<T> for Col<'a, T> {
    type Output = Scaled<T, Col<'a, T>>;

    fn mul(self, rhs: T) -> Scaled<T, Col<'a, T>> {
        Scaled(rhs, self)
    }
}

mul1!(f32, Col<'a, f32>);
mul1!(f64, Col<'a, f64>);
mul1!(Complex<f32>, Col<'a, Complex<f32>>);
mul1!(Complex<f64>, Col<'a, Complex<f64>>);

impl<'a, T> Mul<T> for &'a ColVec<T> {
    type Output = Scaled<T, Col<'a, T>>;

    fn mul(self, rhs: T) -> Scaled<T, Col<'a, T>> {
        Scaled(rhs, self.as_col())
    }
}

mul1!(f32, &'a ColVec<f32>, Col<'a, f32>);
mul1!(f64, &'a ColVec<f64>, Col<'a, f64>);
mul1!(Complex<f32>, &'a ColVec<Complex<f32>>, Col<'a, Complex<f32>>);
mul1!(Complex<f64>, &'a ColVec<Complex<f64>>, Col<'a, Complex<f64>>);

impl<'a, 'b, T> Mul<T> for &'a MutCol<'b, T> {
    type Output = Scaled<T, Col<'a, T>>;

    fn mul(self, rhs: T) -> Scaled<T, Col<'a, T>> {
        Scaled(rhs, *self.as_col())
    }
}

mul2!(f32, &'a MutCol<'b, f32>, Col<'a, f32>);
mul2!(f64, &'a MutCol<'b, f64>, Col<'a, f64>);
mul2!(Complex<f32>, &'a MutCol<'b, Complex<f32>>, Col<'a, Complex<f32>>);
mul2!(Complex<f64>, &'a MutCol<'b, Complex<f64>>, Col<'a, Complex<f64>>);

// mat
impl<'a, T> Mul<T> for &'a Mat<T> {
    type Output = Scaled<T, View<'a, T>>;

    fn mul(self, rhs: T) -> Scaled<T, View<'a, T>> {
        Scaled(rhs, self.as_view())
    }
}

mul1!(f32, &'a Mat<f32>, View<'a, f32>);
mul1!(f64, &'a Mat<f64>, View<'a, f64>);
mul1!(Complex<f32>, &'a Mat<Complex<f32>>, View<'a, Complex<f32>>);
mul1!(Complex<f64>, &'a Mat<Complex<f64>>, View<'a, Complex<f64>>);

impl<'a, 'b, T> Mul<T> for &'a MutView<'b, T> {
    type Output = Scaled<T, View<'a, T>>;

    fn mul(self, rhs: T) -> Scaled<T, View<'a, T>> {
        Scaled(rhs, *self.as_view())
    }
}

mul2!(f32, &'a MutView<'b, f32>, View<'a, f32>);
mul2!(f64, &'a MutView<'b, f64>, View<'a, f64>);
mul2!(Complex<f32>, &'a MutView<'b, Complex<f32>>, View<'a, Complex<f32>>);
mul2!(Complex<f64>, &'a MutView<'b, Complex<f64>>, View<'a, Complex<f64>>);

impl<'a, T> Mul<T> for &'a Trans<Mat<T>> {
    type Output = Scaled<T, Trans<View<'a, T>>>;

    fn mul(self, rhs: T) -> Scaled<T, Trans<View<'a, T>>> {
        Scaled(rhs, Trans(self.0.as_view()))
    }
}

mul1!(f32, &'a Trans<Mat<f32>>, Trans<View<'a, f32>>);
mul1!(f64, &'a Trans<Mat<f64>>, Trans<View<'a, f64>>);
mul1!(Complex<f32>, &'a Trans<Mat<Complex<f32>>>, Trans<View<'a, Complex<f32>>>);
mul1!(Complex<f64>, &'a Trans<Mat<Complex<f64>>>, Trans<View<'a, Complex<f64>>>);

impl<'a, 'b, T> Mul<T> for &'a Trans<MutView<'b, T>> {
    type Output = Scaled<T, Trans<View<'a, T>>>;

    fn mul(self, rhs: T) -> Scaled<T, Trans<View<'a, T>>> {
        Scaled(rhs, Trans(*self.0.as_view()))
    }
}

mul2!(f32, &'a Trans<MutView<'b, f32>>, Trans<View<'a, f32>>);
mul2!(f64, &'a Trans<MutView<'b, f64>>, Trans<View<'a, f64>>);
mul2!(Complex<f32>, &'a Trans<MutView<'b, Complex<f32>>>, Trans<View<'a, Complex<f32>>>);
mul2!(Complex<f64>, &'a Trans<MutView<'b, Complex<f64>>>, Trans<View<'a, Complex<f64>>>);

impl<'a, T> Mul<T> for Trans<View<'a, T>> {
    type Output = Scaled<T, Trans<View<'a, T>>>;

    fn mul(self, rhs: T) -> Scaled<T, Trans<View<'a, T>>> {
        Scaled(rhs, self)
    }
}

mul1!(f32, Trans<View<'a, f32>>);
mul1!(f64, Trans<View<'a, f64>>);
mul1!(Complex<f32>, Trans<View<'a, Complex<f32>>>);
mul1!(Complex<f64>, Trans<View<'a, Complex<f64>>>);

impl<'a, T> Mul<T> for View<'a, T> {
    type Output = Scaled<T, View<'a, T>>;

    fn mul(self, rhs: T) -> Scaled<T, View<'a, T>> {
        Scaled(rhs, self)
    }
}

mul1!(f32, View<'a, f32>);
mul1!(f64, View<'a, f64>);
mul1!(Complex<f32>, View<'a, Complex<f32>>);
mul1!(Complex<f64>, View<'a, Complex<f64>>);

// row
impl<'a, T> Mul<T> for Row<'a, T> {
    type Output = Scaled<T, Row<'a, T>>;

    fn mul(self, rhs: T) -> Scaled<T, Row<'a, T>> {
        Scaled(rhs, self)
    }
}

mul1!(f32, Row<'a, f32>);
mul1!(f64, Row<'a, f64>);
mul1!(Complex<f32>, Row<'a, Complex<f32>>);
mul1!(Complex<f64>, Row<'a, Complex<f64>>);

impl<'a, T> Mul<T> for &'a RowVec<T> {
    type Output = Scaled<T, Row<'a, T>>;

    fn mul(self, rhs: T) -> Scaled<T, Row<'a, T>> {
        Scaled(rhs, self.as_row())
    }
}

mul1!(f32, &'a RowVec<f32>, Row<'a, f32>);
mul1!(f64, &'a RowVec<f64>, Row<'a, f64>);
mul1!(Complex<f32>, &'a RowVec<Complex<f32>>, Row<'a, Complex<f32>>);
mul1!(Complex<f64>, &'a RowVec<Complex<f64>>, Row<'a, Complex<f64>>);

impl<'a, 'b, T> Mul<T> for &'a MutRow<'b, T> {
    type Output = Scaled<T, Row<'a, T>>;

    fn mul(self, rhs: T) -> Scaled<T, Row<'a, T>> {
        Scaled(rhs, *self.as_row())
    }
}

mul2!(f32, &'a MutRow<'b, f32>, Row<'a, f32>);
mul2!(f64, &'a MutRow<'b, f64>, Row<'a, f64>);
mul2!(Complex<f32>, &'a MutRow<'b, Complex<f32>>, Row<'a, Complex<f32>>);
mul2!(Complex<f64>, &'a MutRow<'b, Complex<f64>>, Row<'a, Complex<f64>>);

// scaled
impl<T, M> Mul<T> for Scaled<T, M> where T: Mul<Output=T> {
    type Output = Scaled<T, M>;

    fn mul(self, rhs: T) -> Scaled<T, M> {
        Scaled(self.0 * rhs, self.1)
    }
}

macro_rules! mul {
    ($($ty:ty),+) => {
        $(
            impl<M> Mul<Scaled<$ty, M>> for $ty {
                type Output = Scaled<$ty, M>;

                fn mul(self, rhs: Scaled<$ty, M>) -> Scaled<$ty, M> {
                    rhs * self
                }
            }
        )+
    }
}

mul!(f32, f64, Complex<f32>, Complex<f64>);
