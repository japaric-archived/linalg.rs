use std::ops::{Sub, Neg};

use assign::SubAssign;
use blas::Axpy;
use onezero::One;

use {Col, ColVec, Mat, MutCol, MutRow, MutView, Row, RowVec, Scaled, Trans, View};

macro_rules! sub0 {
    ($lhs:ty, $rhs:ty) => {
        impl<T> Sub<$rhs> for $lhs where T: Axpy + Neg<Output=T> + One {
            type Output = $lhs;

            fn sub(mut self, rhs: $rhs) -> $lhs {
                self.sub_assign(&rhs);
                self
            }
        }
    }
}

macro_rules! sub1 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, T> Sub<$rhs> for $lhs where T: Axpy + Neg<Output=T> + One {
            type Output = $lhs;

            fn sub(mut self, rhs: $rhs) -> $lhs {
                self.sub_assign(&rhs);
                self
            }
        }
    }
}

macro_rules! sub1_ {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, T> Sub<$rhs> for $lhs where T: Axpy + Neg<Output=T> + One {
            type Output = $lhs;

            fn sub(mut self, rhs: $rhs) -> $lhs {
                self.sub_assign(rhs);
                self
            }
        }
    }
}

macro_rules! sub1c {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, T> Sub<$rhs> for $lhs where T: 'a + Axpy + Neg<Output=T> + One + Clone {
            type Output = $lhs;

            fn sub(mut self, rhs: $rhs) -> $lhs {
                self.sub_assign(&rhs);
                self
            }
        }
    }
}

macro_rules! sub2 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T> Sub<$rhs> for $lhs where T: Axpy + Neg<Output=T> + One {
            type Output = $lhs;

            fn sub(mut self, rhs: $rhs) -> $lhs {
                self.sub_assign(&rhs);
                self
            }
        }
    }
}

macro_rules! sub2_ {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T> Sub<$rhs> for $lhs where T: Axpy + Neg<Output=T> + One {
            type Output = $lhs;

            fn sub(mut self, rhs: $rhs) -> $lhs {
                self.sub_assign(rhs);
                self
            }
        }
    }
}

macro_rules! sub2c {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T> Sub<$rhs> for $lhs where T: 'b + Axpy + Clone + Neg<Output=T> + One {
            type Output = $lhs;

            fn sub(mut self, rhs: $rhs) -> $lhs {
                self.sub_assign(&rhs);
                self
            }
        }
    }
}

macro_rules! sub3 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, 'c, T> Sub<$rhs> for $lhs where T: Axpy + Neg<Output=T> + One {
            type Output = $lhs;

            fn sub(mut self, rhs: $rhs) -> $lhs {
                self.sub_assign(rhs);
                self
            }
        }
    }
}

// col
sub0!(ColVec<T>, T);
sub1!(ColVec<T>, Col<'a, T>);
sub1_!(ColVec<T>, &'a ColVec<T>);
sub2_!(ColVec<T>, &'a MutCol<'b, T>);
sub1c!(ColVec<T>, Scaled<T, Col<'a, T>>);

sub1!(MutCol<'a, T>, T);
sub2c!(MutCol<'a, T>, Col<'b, T>);
sub2_!(MutCol<'a, T>, &'b ColVec<T>);
sub3!(MutCol<'a, T>, &'b MutCol<'c, T>);
sub2c!(MutCol<'a, T>, Scaled<T, Col<'b, T>>);

// mat
sub0!(Mat<T>, T);
sub1_!(Mat<T>, &'a Mat<T>);
sub2_!(Mat<T>, &'a MutView<'b, T>);
sub1c!(Mat<T>, Scaled<T, View<'a, T>>);
sub1_!(Mat<T>, &'a Trans<Mat<T>>);
sub2_!(Mat<T>, &'a Trans<MutView<'b, T>>);
sub1!(Mat<T>, Trans<View<'a, T>>);
sub1!(Mat<T>, View<'a, T>);

sub1!(MutView<'a, T>, T);
sub2_!(MutView<'a, T>, &'b Mat<T>);
sub3!(MutView<'a, T>, &'b MutView<'c, T>);
sub2c!(MutView<'a, T>, Scaled<T, View<'b, T>>);
sub2_!(MutView<'a, T>, &'b Trans<Mat<T>>);
sub3!(MutView<'a, T>, &'b Trans<MutView<'b, T>>);
sub2!(MutView<'a, T>, Trans<View<'b, T>>);
sub2!(MutView<'a, T>, View<'b, T>);

sub0!(Trans<Mat<T>>, T);
sub1_!(Trans<Mat<T>>, &'a Mat<T>);
sub2_!(Trans<Mat<T>>, &'a MutView<'b, T>);
sub1c!(Trans<Mat<T>>, Scaled<T, View<'a, T>>);
sub1_!(Trans<Mat<T>>, &'a Trans<Mat<T>>);
sub2_!(Trans<Mat<T>>, &'a Trans<MutView<'b, T>>);
sub1!(Trans<Mat<T>>, Trans<View<'a, T>>);
sub1!(Trans<Mat<T>>, View<'a, T>);

sub1!(Trans<MutView<'a, T>>, T);
sub2_!(Trans<MutView<'a, T>>, &'b Mat<T>);
sub3!(Trans<MutView<'a, T>>, &'b MutView<'c, T>);
sub2c!(Trans<MutView<'a, T>>, Scaled<T, View<'b, T>>);
sub2_!(Trans<MutView<'a, T>>, &'b Trans<Mat<T>>);
sub3!(Trans<MutView<'a, T>>, &'b Trans<MutView<'b, T>>);
sub2!(Trans<MutView<'a, T>>, Trans<View<'b, T>>);
sub2!(Trans<MutView<'a, T>>, View<'b, T>);

// row
sub0!(RowVec<T>, T);
sub2_!(RowVec<T>, &'a MutRow<'b, T>);
sub1!(RowVec<T>, Row<'a, T>);
sub1_!(RowVec<T>, &'a RowVec<T>);
sub1c!(RowVec<T>, Scaled<T, Row<'a, T>>);

sub1!(MutRow<'a, T>, T);
sub3!(MutRow<'a, T>, &'b MutRow<'c, T>);
sub2c!(MutRow<'a, T>, Row<'b, T>);
sub2_!(MutRow<'a, T>, &'b RowVec<T>);
sub2c!(MutRow<'a, T>, Scaled<T, Row<'b, T>>);
