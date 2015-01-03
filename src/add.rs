use onezero::One;
use std::ops::Add;

use {Col, ColVec, Mat, MutCol, MutRow, MutView, Row, RowVec, Scaled, Trans, View};
use blas::axpy::Axpy;
use traits::AddAssign;

macro_rules! add0 {
    ($lhs:ty, $rhs:ty) => {
        impl<T> Add<$rhs, $lhs> for $lhs where T: Axpy + One {
            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<T> Add<$lhs, $lhs> for $rhs where T: Axpy + One {
            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

macro_rules! add1 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, T> Add<$rhs, $lhs> for $lhs where T: Axpy + One {
            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<'a, T> Add<$lhs, $lhs> for $rhs where T: Axpy + One {
            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

macro_rules! add1c {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, T> Add<$rhs, $lhs> for $lhs where T: Axpy + One + Clone {
            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<'a, T> Add<$lhs, $lhs> for $rhs where T: Axpy + One + Clone {
            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

macro_rules! add2 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T> Add<$rhs, $lhs> for $lhs where T: Axpy + One {
            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<'a, 'b, T> Add<$lhs, $lhs> for $rhs where T: Axpy + One {
            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

macro_rules! add2c {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T> Add<$rhs, $lhs> for $lhs where T: Axpy + Clone + One {
            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<'a, 'b, T> Add<$lhs, $lhs> for $rhs where T: Axpy + Clone + One {
            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

macro_rules! add3 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, 'c, T> Add<$rhs, $lhs> for $lhs where T: Axpy + One {
            fn add(mut self, rhs: $rhs) -> $lhs {
                self.add_assign(rhs);
                self
            }
        }

        impl<'a, 'b, 'c, T> Add<$lhs, $lhs> for $rhs where T: Axpy + One {
            fn add(self, rhs: $lhs) -> $lhs {
                rhs + self
            }
        }
    }
}

// col
add0!(ColVec<T>, T);
add1!(ColVec<T>, Col<'a, T>);
add1!(ColVec<T>, &'a ColVec<T>);
add2!(ColVec<T>, &'a MutCol<'b, T>);
add1!(ColVec<T>, Scaled<T, Col<'a, T>>);

add1!(MutCol<'a, T>, T);
add2!(MutCol<'a, T>, Col<'a, T>);
add2!(MutCol<'a, T>, &'a ColVec<T>);
add3!(MutCol<'a, T>, &'a MutCol<'b, T>);
add2!(MutCol<'a, T>, Scaled<T, Col<'a, T>>);

// mat
add0!(Mat<T>, T);
add1!(Mat<T>, &'a Mat<T>);
add2!(Mat<T>, &'a MutView<'b, T>);
add1c!(Mat<T>, Scaled<T, View<'a, T>>);
add1!(Mat<T>, &'a Trans<Mat<T>>);
add2!(Mat<T>, &'a Trans<MutView<'b, T>>);
add1!(Mat<T>, Trans<View<'a, T>>);
add1!(Mat<T>, View<'a, T>);

add1c!(MutView<'a, T>, T);
add2!(MutView<'a, T>, &'b Mat<T>);
add3!(MutView<'a, T>, &'b MutView<'c, T>);
add2c!(MutView<'a, T>, Scaled<T, View<'b, T>>);
add2!(MutView<'a, T>, &'b Trans<Mat<T>>);
add3!(MutView<'a, T>, &'b Trans<MutView<'b, T>>);
add2!(MutView<'a, T>, Trans<View<'b, T>>);
add2!(MutView<'a, T>, View<'b, T>);

add0!(Trans<Mat<T>>, T);
add1!(Trans<Mat<T>>, &'a Mat<T>);
add2!(Trans<Mat<T>>, &'a MutView<'b, T>);
add1c!(Trans<Mat<T>>, Scaled<T, View<'a, T>>);
add1!(Trans<Mat<T>>, &'a Trans<Mat<T>>);
add2!(Trans<Mat<T>>, &'a Trans<MutView<'b, T>>);
add1!(Trans<Mat<T>>, Trans<View<'a, T>>);
add1!(Trans<Mat<T>>, View<'a, T>);

add1c!(Trans<MutView<'a, T>>, T);
add2!(Trans<MutView<'a, T>>, &'b Mat<T>);
add3!(Trans<MutView<'a, T>>, &'b MutView<'c, T>);
add2c!(Trans<MutView<'a, T>>, Scaled<T, View<'b, T>>);
add2!(Trans<MutView<'a, T>>, &'b Trans<Mat<T>>);
add3!(Trans<MutView<'a, T>>, &'b Trans<MutView<'b, T>>);
add2!(Trans<MutView<'a, T>>, Trans<View<'b, T>>);
add2!(Trans<MutView<'a, T>>, View<'b, T>);

// row
add0!(RowVec<T>, T);
add2!(RowVec<T>, &'a MutRow<'b, T>);
add1!(RowVec<T>, Row<'a, T>);
add1!(RowVec<T>, &'a RowVec<T>);
add1!(RowVec<T>, Scaled<T, Row<'a, T>>);

add1!(MutRow<'a, T>, T);
add3!(MutRow<'a, T>, &'a MutRow<'b, T>);
add2!(MutRow<'a, T>, Row<'a, T>);
add2!(MutRow<'a, T>, &'a RowVec<T>);
add2c!(MutRow<'a, T>, Scaled<T, Row<'a, T>>);
