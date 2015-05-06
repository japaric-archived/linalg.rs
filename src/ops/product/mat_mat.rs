use std::ops::Mul;

use blas::Transpose;

use Forward;
use traits::{Matrix, Slice};
use {Chain, Mat, Scaled, Transposed, SubMat, SubMatMut};

// LHS: Chain, &Mat, Scaled<Chain>, Scaled<Transposed<SubMat>>, Scaled<SubMat>, &Transposed<Mat>,
// Transposed<SubMat>, &Transposed<SubMatMut>, SubMat, &SubMatMut
// RHS: Same as RHS
//
// -> 100 implementations

// 9 impls
// Core implementations
impl<'a, T> Mul<Chain<'a, T>> for Chain<'a, T> {
    type Output = Chain<'a, T>;

    fn mul(self, rhs: Chain<'a, T>) -> Chain<'a, T> {
        assert_eq_inner_dimensions!(self, rhs);

        let Chain { first, second, mut tail } = self;

        tail.reserve(rhs.len());

        tail.push(rhs.first);
        tail.push(rhs.second);
        tail.push_all(&rhs.tail);

        Chain {
            first: first,
            second: second,
            tail: tail,
        }
    }
}

impl<'a, T> Mul<Transposed<SubMat<'a, T>>> for Chain<'a, T> {
    type Output = Chain<'a, T>;

    fn mul(mut self, rhs: Transposed<SubMat<'a, T>>) -> Chain<'a, T> {
        assert_eq_inner_dimensions!(self, rhs);

        self.tail.push((Transpose::Yes, rhs.0));
        self
    }
}

impl<'a, T> Mul<SubMat<'a, T>> for Chain<'a, T> {
    type Output = Chain<'a, T>;

    fn mul(mut self, rhs: SubMat<'a, T>) -> Chain<'a, T> {
        assert_eq_inner_dimensions!(self, rhs);

        self.tail.push((Transpose::No, rhs));
        self
    }
}

impl<'a, T> Mul<Chain<'a, T>> for Transposed<SubMat<'a, T>> {
    type Output = Chain<'a, T>;

    fn mul(self, rhs: Chain<'a, T>) -> Chain<'a, T> {
        assert_eq_inner_dimensions!(self, rhs);

        let Chain { first: was_first, second: was_second, mut tail } = rhs;

        tail.insert(0, was_second);

        Chain {
            first: (Transpose::Yes, self.0),
            second: was_first,
            tail: tail,
        }
    }
}

impl<'a, T> Mul<Transposed<SubMat<'a, T>>> for Transposed<SubMat<'a, T>> {
    type Output = Chain<'a, T>;

    fn mul(self, rhs: Transposed<SubMat<'a, T>>) -> Chain<'a, T> {
        assert_eq_inner_dimensions!(self, rhs);

        Chain {
            first: (Transpose::Yes, self.0),
            second: (Transpose::Yes, rhs.0),
            tail: vec![],
        }
    }
}

impl<'a, T> Mul<SubMat<'a, T>> for Transposed<SubMat<'a, T>> {
    type Output = Chain<'a, T>;

    fn mul(self, rhs: SubMat<'a, T>) -> Chain<'a, T> {
        assert_eq_inner_dimensions!(self, rhs);

        Chain {
            first: (Transpose::Yes, self.0),
            second: (Transpose::No, rhs),
            tail: vec![],
        }
    }
}

impl<'a, T> Mul<Chain<'a, T>> for SubMat<'a, T> {
    type Output = Chain<'a, T>;

    fn mul(self, rhs: Chain<'a, T>) -> Chain<'a, T> {
        assert_eq_inner_dimensions!(self, rhs);

        let Chain { first: was_first, second: was_second, mut tail } = rhs;

        tail.insert(0, was_second);

        Chain {
            first: (Transpose::No, self),
            second: was_first,
            tail: tail,
        }
    }
}

impl<'a, T> Mul<Transposed<SubMat<'a, T>>> for SubMat<'a, T> {
    type Output = Chain<'a, T>;

    fn mul(self, rhs: Transposed<SubMat<'a, T>>) -> Chain<'a, T> {
        assert_eq_inner_dimensions!(self, rhs);

        Chain {
            first: (Transpose::No, self),
            second: (Transpose::Yes, rhs.0),
            tail: vec![],
        }
    }
}

impl<'a, T> Mul<SubMat<'a, T>> for SubMat<'a, T> {
    type Output = Chain<'a, T>;

    fn mul(self, rhs: SubMat<'a, T>) -> Chain<'a, T> {
        assert_eq_inner_dimensions!(self, rhs);

        Chain {
            first: (Transpose::No, self),
            second: (Transpose::No, rhs),
            tail: vec![],
        }
    }
}

// Secondary implementations
macro_rules! scaled {
    ($lhs:ty { $($rhs:ty),+ }) => {
        $(
            impl<'a, T> Mul<$rhs> for Scaled<$lhs> {
                type Output = Scaled<Chain<'a, T>>;

                fn mul(self, rhs: $rhs) -> Scaled<Chain<'a, T>> {
                    Scaled(self.0, self.1 * rhs)
                }
            }

            impl<'a, T> Mul<Scaled<$rhs>> for $lhs {
                type Output = Scaled<Chain<'a, T>>;

                fn mul(self, rhs: Scaled<$rhs>) -> Scaled<Chain<'a, T>> {
                    Scaled(rhs.0, self * rhs.1)
                }
            }

            impl<'a, T> Mul<Scaled<$rhs>> for Scaled<$lhs> where T: Mul<Output=T> {
                type Output = Scaled<Chain<'a, T>>;

                fn mul(self, rhs: Scaled<$rhs>) -> Scaled<Chain<'a, T>> {
                    Scaled(self.0 * rhs.0, self.1 * rhs.1)
                }
            }
         )+
    }
}

// 9 impls
scaled!(Chain<'a, T> { Chain<'a, T>, Transposed<SubMat<'a, T>>, SubMat<'a, T> });

// 9 impls
scaled!(Transposed<SubMat<'a, T>> { Chain<'a, T>, Transposed<SubMat<'a, T>>, SubMat<'a, T> });

// 9 impls
scaled!(SubMat<'a, T> { Chain<'a, T>, Transposed<SubMat<'a, T>>, SubMat<'a, T> });

macro_rules! forward {
    ($lhs:ty { $($rhs:ty => $output:ty),+, }) => {
        $(
            impl<'a, 'b, 'c, T> Mul<$rhs> for $lhs {
                type Output = $output;

                fn mul(self, rhs: $rhs) -> $output {
                    self.slice(..) * rhs.slice(..)
                }
            }
         )+
    }
}

// 4 impls
forward!(Chain<'a, T> {
    &'a Mat<T>
        => Chain<'a, T>,

    &'a Transposed<Mat<T>>
        => Chain<'a, T>,

    &'a Transposed<SubMatMut<'b, T>>
        => Chain<'a, T>,

    &'a SubMatMut<'b, T>
        => Chain<'a, T>,
});

// 10 impls
forward!(&'a Mat<T> {
    Chain<'a, T>
        => Chain<'a, T>,

    &'a Mat<T>
        => Chain<'a, T>,

    Scaled<Chain<'a, T>>
        => Scaled<Chain<'a, T>>,

    Scaled<Transposed<SubMat<'a, T>>>
        => Scaled<Chain<'a, T>>,

    Scaled<SubMat<'a, T>>
        => Scaled<Chain<'a, T>>,

    &'a Transposed<Mat<T>>
        => Chain<'a, T>,

    Transposed<SubMat<'a, T>>
        => Chain<'a, T>,

    &'a Transposed<SubMatMut<'b, T>>
        => Chain<'a, T>,

    SubMat<'a, T>
        => Chain<'a, T>,

    &'a SubMatMut<'b, T>
        => Chain<'a, T>,
});

// 4 impls
forward!(Scaled<Chain<'a, T>> {
    &'a Mat<T>
        => Scaled<Chain<'a, T>>,

    &'a Transposed<Mat<T>>
        => Scaled<Chain<'a, T>>,

    &'a Transposed<SubMatMut<'b, T>>
        => Scaled<Chain<'a, T>>,

    &'a SubMatMut<'b, T>
        => Scaled<Chain<'a, T>>,
});

// 4 impls
forward!(Scaled<Transposed<SubMat<'a, T>>> {
    &'a Mat<T>
        => Scaled<Chain<'a, T>>,

    &'a Transposed<Mat<T>>
        => Scaled<Chain<'a, T>>,

    &'a Transposed<SubMatMut<'b, T>>
        => Scaled<Chain<'a, T>>,

    &'a SubMatMut<'b, T>
        => Scaled<Chain<'a, T>>,
});

// 4 impls
forward!(Scaled<SubMat<'a, T>> {
    &'a Mat<T>
        => Scaled<Chain<'a, T>>,

    &'a Transposed<Mat<T>>
        => Scaled<Chain<'a, T>>,

    &'a Transposed<SubMatMut<'b, T>>
        => Scaled<Chain<'a, T>>,

    &'a SubMatMut<'b, T>
        => Scaled<Chain<'a, T>>,
});

// 10 impls
forward!(&'a Transposed<Mat<T>> {
    Chain<'a, T>
        => Chain<'a, T>,

    &'a Mat<T>
        => Chain<'a, T>,

    Scaled<Chain<'a, T>>
        => Scaled<Chain<'a, T>>,

    Scaled<Transposed<SubMat<'a, T>>>
        => Scaled<Chain<'a, T>>,

    Scaled<SubMat<'a, T>>
        => Scaled<Chain<'a, T>>,

    &'a Transposed<Mat<T>>
        => Chain<'a, T>,

    Transposed<SubMat<'a, T>>
        => Chain<'a, T>,

    &'a Transposed<SubMatMut<'b, T>>
        => Chain<'a, T>,

    SubMat<'a, T>
        => Chain<'a, T>,

    &'a SubMatMut<'b, T>
        => Chain<'a, T>,
});

// 4 impls
forward!(Transposed<SubMat<'a, T>> {
    &'a Mat<T>
        => Chain<'a, T>,

    &'a Transposed<Mat<T>>
        => Chain<'a, T>,

    &'a Transposed<SubMatMut<'b, T>>
        => Chain<'a, T>,

    &'a SubMatMut<'b, T>
        => Chain<'a, T>,
});

// 10 impls
forward!(&'a Transposed<SubMatMut<'b, T>> {
    Chain<'a, T>
        => Chain<'a, T>,

    &'a Mat<T>
        => Chain<'a, T>,

    Scaled<Chain<'a, T>>
        => Scaled<Chain<'a, T>>,

    Scaled<Transposed<SubMat<'a, T>>>
        => Scaled<Chain<'a, T>>,

    Scaled<SubMat<'a, T>>
        => Scaled<Chain<'a, T>>,

    &'a Transposed<Mat<T>>
        => Chain<'a, T>,

    Transposed<SubMat<'a, T>>
        => Chain<'a, T>,

    &'a Transposed<SubMatMut<'c, T>>
        => Chain<'a, T>,

    SubMat<'a, T>
        => Chain<'a, T>,

    &'a SubMatMut<'b, T>
        => Chain<'a, T>,
});

// 4 impls
forward!(SubMat<'a, T> {
    &'a Mat<T>
        => Chain<'a, T>,

    &'a Transposed<Mat<T>>
        => Chain<'a, T>,

    &'a Transposed<SubMatMut<'b, T>>
        => Chain<'a, T>,

    &'a SubMatMut<'b, T>
        => Chain<'a, T>,
});

// 10 impls
forward!(&'a SubMatMut<'b, T> {
    Chain<'a, T>
        => Chain<'a, T>,

    &'a Mat<T>
        => Chain<'a, T>,

    Scaled<Chain<'a, T>>
        => Scaled<Chain<'a, T>>,

    Scaled<Transposed<SubMat<'a, T>>>
        => Scaled<Chain<'a, T>>,

    Scaled<SubMat<'a, T>>
        => Scaled<Chain<'a, T>>,

    &'a Transposed<Mat<T>>
        => Chain<'a, T>,

    Transposed<SubMat<'a, T>>
        => Chain<'a, T>,

    &'a Transposed<SubMatMut<'c, T>>
        => Chain<'a, T>,

    SubMat<'a, T>
        => Chain<'a, T>,

    &'a SubMatMut<'b, T>
        => Chain<'a, T>,
});
