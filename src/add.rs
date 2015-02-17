use complex::Complex;
use onezero::One;
use std::ops::Add;

use {Col, ColVec, Mat, MutCol, MutRow, MutView, Row, RowVec, Scaled, Trans, View};
use blas::axpy::Axpy;
use traits::AddAssign;

macro_rules! add0 {
    ($lhs:ty, $rhs:ty) => {
        impl<T> Add<$rhs> for $lhs where T: Axpy + One {
            type Output = $lhs;

            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<T> Add<$lhs> for $rhs where T: Axpy + One {
            type Output = $lhs;

            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

macro_rules! add0_ {
    ($lhs:ty, $rhs:ty) => {
        impl Add<$rhs> for $lhs {
            type Output = $rhs;

            fn add(self, rhs: $rhs) -> $rhs {
                rhs + self
            }
        }
    };
}

macro_rules! add1 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, T> Add<$rhs> for $lhs where T: Axpy + One {
            type Output = $lhs;

            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<'a, T> Add<$lhs> for $rhs where T: Axpy + One {
            type Output = $lhs;

            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

macro_rules! add1_ {
    ($lhs:ty, $rhs:ty) => {
        impl<'a> Add<$rhs> for $lhs {
            type Output = $rhs;

            fn add(self, rhs: $rhs) -> $rhs {
                rhs + self
            }
        }
    };
}

macro_rules! add1c {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, T> Add<$rhs> for $lhs where T: Axpy + One + Clone {
            type Output = $lhs;

            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<'a, T> Add<$lhs> for $rhs where T: Axpy + One + Clone {
            type Output = $lhs;

            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

macro_rules! add2 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T> Add<$rhs> for $lhs where T: Axpy + One {
            type Output = $lhs;

            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<'a, 'b, T> Add<$lhs> for $rhs where T: Axpy + One {
            type Output = $lhs;

            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

macro_rules! add2c {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T> Add<$rhs> for $lhs where T: Axpy + Clone + One {
            type Output = $lhs;

            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<'a, 'b, T> Add<$lhs> for $rhs where T: Axpy + Clone + One {
            type Output = $lhs;

            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

macro_rules! add3 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, 'c, T> Add<$rhs> for $lhs where T: Axpy + One {
            type Output = $lhs;

            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<'a, 'b, 'c, T> Add<$lhs> for $rhs where T: Axpy + One {
            type Output = $lhs;

            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

// col
add1!(ColVec<T>, Col<'a, T>);
add1!(ColVec<T>, &'a ColVec<T>);
add2!(ColVec<T>, &'a MutCol<'b, T>);
add1!(ColVec<T>, Scaled<T, Col<'a, T>>);

impl<T> Add<T> for ColVec<T> where T: Axpy + One {
    type Output = ColVec<T>;

    fn add(mut self, rhs: T) -> ColVec<T> {
        self.add_assign(rhs);
        self
    }
}

add0_!(f32, ColVec<f32>);
add0_!(f64, ColVec<f64>);
add0_!(Complex<f32>, ColVec<Complex<f32>>);
add0_!(Complex<f64>, ColVec<Complex<f64>>);

add2!(MutCol<'a, T>, Col<'a, T>);
add2!(MutCol<'a, T>, &'a ColVec<T>);
add3!(MutCol<'a, T>, &'a MutCol<'b, T>);
add2!(MutCol<'a, T>, Scaled<T, Col<'a, T>>);

impl<'a, T> Add<T> for MutCol<'a, T> where T: Axpy + One {
    type Output = MutCol<'a, T>;

    fn add(mut self, rhs: T) -> MutCol<'a, T> {
        self.add_assign(rhs);
        self
    }
}

add1_!(f32, MutCol<'a, f32>);
add1_!(f64, MutCol<'a, f64>);
add1_!(Complex<f32>, MutCol<'a, Complex<f32>>);
add1_!(Complex<f64>, MutCol<'a, Complex<f64>>);

// mat
add1!(Mat<T>, &'a Mat<T>);
add2!(Mat<T>, &'a MutView<'b, T>);
add1c!(Mat<T>, Scaled<T, View<'a, T>>);
add1!(Mat<T>, &'a Trans<Mat<T>>);
add2!(Mat<T>, &'a Trans<MutView<'b, T>>);
add1!(Mat<T>, Trans<View<'a, T>>);
add1!(Mat<T>, View<'a, T>);

impl<T> Add<T> for Mat<T> where T: Axpy + One {
    type Output = Mat<T>;

    fn add(mut self, rhs: T) -> Mat<T> {
        self.add_assign(rhs);
        self
    }
}

add0_!(f32, Mat<f32>);
add0_!(f64, Mat<f64>);
add0_!(Complex<f32>, Mat<Complex<f32>>);
add0_!(Complex<f64>, Mat<Complex<f64>>);

add2!(MutView<'a, T>, &'b Mat<T>);
add3!(MutView<'a, T>, &'b MutView<'c, T>);
add2c!(MutView<'a, T>, Scaled<T, View<'b, T>>);
add2!(MutView<'a, T>, &'b Trans<Mat<T>>);
add3!(MutView<'a, T>, &'b Trans<MutView<'b, T>>);
add2!(MutView<'a, T>, Trans<View<'b, T>>);
add2!(MutView<'a, T>, View<'b, T>);

impl<'a, T> Add<T> for MutView<'a, T> where T: Axpy + Clone + One {
    type Output = MutView<'a, T>;

    fn add(mut self, rhs: T) -> MutView<'a, T> {
        self.add_assign(rhs);
        self
    }
}

add1_!(f32, MutView<'a, f32>);
add1_!(f64, MutView<'a, f64>);
add1_!(Complex<f32>, MutView<'a, Complex<f32>>);
add1_!(Complex<f64>, MutView<'a, Complex<f64>>);

add1!(Trans<Mat<T>>, &'a Mat<T>);
add2!(Trans<Mat<T>>, &'a MutView<'b, T>);
add1c!(Trans<Mat<T>>, Scaled<T, View<'a, T>>);
add1!(Trans<Mat<T>>, &'a Trans<Mat<T>>);
add2!(Trans<Mat<T>>, &'a Trans<MutView<'b, T>>);
add1!(Trans<Mat<T>>, Trans<View<'a, T>>);
add1!(Trans<Mat<T>>, View<'a, T>);

impl<T> Add<T> for Trans<Mat<T>> where T: Axpy + One {
    type Output = Trans<Mat<T>>;

    fn add(mut self, rhs: T) -> Trans<Mat<T>> {
        self.add_assign(rhs);
        self
    }
}

add0_!(f32, Trans<Mat<f32>>);
add0_!(f64, Trans<Mat<f64>>);
add0_!(Complex<f32>, Trans<Mat<Complex<f32>>>);
add0_!(Complex<f64>, Trans<Mat<Complex<f64>>>);

add2!(Trans<MutView<'a, T>>, &'b Mat<T>);
add3!(Trans<MutView<'a, T>>, &'b MutView<'c, T>);
add2c!(Trans<MutView<'a, T>>, Scaled<T, View<'b, T>>);
add2!(Trans<MutView<'a, T>>, &'b Trans<Mat<T>>);
add3!(Trans<MutView<'a, T>>, &'b Trans<MutView<'b, T>>);
add2!(Trans<MutView<'a, T>>, Trans<View<'b, T>>);
add2!(Trans<MutView<'a, T>>, View<'b, T>);

impl<'a, T> Add<T> for Trans<MutView<'a, T>> where T: Axpy + Clone + One {
    type Output = Trans<MutView<'a, T>>;

    fn add(mut self, rhs: T) -> Trans<MutView<'a, T>> {
        self.add_assign(rhs);
        self
    }
}

add1_!(f32, Trans<MutView<'a, f32>>);
add1_!(f64, Trans<MutView<'a, f64>>);
add1_!(Complex<f32>, Trans<MutView<'a, Complex<f32>>>);
add1_!(Complex<f64>, Trans<MutView<'a, Complex<f64>>>);

// row
add2!(RowVec<T>, &'a MutRow<'b, T>);
add1!(RowVec<T>, Row<'a, T>);
add1!(RowVec<T>, &'a RowVec<T>);
add1!(RowVec<T>, Scaled<T, Row<'a, T>>);

impl<T> Add<T> for RowVec<T> where T: Axpy + One {
    type Output = RowVec<T>;

    fn add(mut self, rhs: T) -> RowVec<T> {
        self.add_assign(rhs);
        self
    }
}

add0_!(f32, RowVec<f32>);
add0_!(f64, RowVec<f64>);
add0_!(Complex<f32>, RowVec<Complex<f32>>);
add0_!(Complex<f64>, RowVec<Complex<f64>>);

add3!(MutRow<'a, T>, &'a MutRow<'b, T>);
add2!(MutRow<'a, T>, Row<'a, T>);
add2!(MutRow<'a, T>, &'a RowVec<T>);
add2c!(MutRow<'a, T>, Scaled<T, Row<'a, T>>);

impl<'a, T> Add<T> for MutRow<'a, T> where T: Axpy + One {
    type Output = MutRow<'a, T>;

    fn add(mut self, rhs: T) -> MutRow<'a, T> {
        self.add_assign(rhs);
        self
    }
}

add1_!(f32, MutRow<'a, f32>);
add1_!(f64, MutRow<'a, f64>);
add1_!(Complex<f32>, MutRow<'a, Complex<f32>>);
add1_!(Complex<f64>, MutRow<'a, Complex<f64>>);
