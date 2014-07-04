use num::Complex;
use quickcheck::{TestResult,quickcheck};
use rand::distributions::{IndependentSample,Range};
use std::iter::AdditiveIterator;
use std::{num,rand};

use array::traits::{ArrayNorm2,ArrayScale,ArrayShape};
use mat;
use test::tol;
// FIXME mozilla/rust#5992 Use std {Add,Mul,Sub}Assign
// FIXME mozilla/rust#6515 Use std Index
use traits::{AddAssign,Index,Iterable,MulAssign,SubAssign};

mod col;
mod diag;
mod row;
mod view;

// mat
#[test]
fn from_elem() {
    fn prop(nrows: uint, ncols: uint, elem: f32) -> bool {
        let m = mat::from_elem((nrows, ncols), elem);

        m.shape() == (nrows, ncols) &&
            m.unwrap() == Vec::from_elem(nrows * ncols, elem)
    }

    quickcheck(prop);
}

#[test]
fn from_fn() {
    fn prop(nrows: uint, ncols: uint) -> bool {
        let m = mat::from_fn((nrows, ncols), |i, j| i + j);

        m.shape() == (nrows, ncols) &&
            m.unwrap() == Vec::from_fn(nrows * ncols, |i| {
                let r = i / ncols;
                let c = i % ncols;
                r + c
            })
    }

    quickcheck(prop);
}

#[test]
fn map() {
    fn prop(shape: (uint, uint), (low, high): (f32, f32)) -> TestResult {
        if low >= high {
            return TestResult::discard();
        }

        let between = Range::new(low, high);
        let mut rng = rand::task_rng();
        let rng = &mut rng;

        let xs = mat::rand(shape, &between, rng);
        let mut ys = xs.clone();
        ys.map(|y| y.sin());

        let (nrows, ncols) = shape;
        let (rows, cols) = (range(0, nrows), range(0, ncols));
        TestResult::from_bool(rows.zip(cols).all(|ref i| {
            xs.index(i).sin().eq(ys.index(i))
        }))
    }

    quickcheck(prop);
}

#[test]
fn rand() {
    fn prop(shape: (uint, uint), (low, high): (f32, f32)) -> TestResult {
        if low >= high {
            return TestResult::discard();
        }

        let between = Range::new(low, high);
        let mut rng = rand::task_rng();
        let rng = &mut rng;
        let v = mat::rand(shape, &between, rng);

        TestResult::from_bool(v.shape() == shape &&
                              v.all(|&e| e >= low && e <= high))
    }

    quickcheck(prop);
}

macro_rules! op_assign {
    ($name:ident, $ty:ty, $op:ident, $op_assign:ident) => {
        #[test]
        fn $name() {
            fn prop(shape: (uint, uint),
                    (low, high): ($ty, $ty)) -> TestResult {
                if low >= high {
                    return TestResult::discard();
                }

                let between = Range::new(low, high);
                let mut rng = rand::task_rng();
                let rng = &mut rng;

                let xs = mat::rand(shape, &between, rng);
                let ys = mat::rand(shape, &between, rng);
                let mut zs = xs.clone();

                zs.$op_assign(&ys);

                let (nrows, ncols) = shape;
                TestResult::from_bool(xs.shape() == zs.shape() &&
                    range(0, nrows).zip(range(0, ncols)).all(|ref i| {
                        xs.index(i).$op(ys.index(i)).eq(zs.index(i))
                    }))
            }

            quickcheck(prop);
        }
    }
}

macro_rules! op_assign_complex {
    ($name:ident, $ty:ty, $op:ident, $op_assign:ident) => {
        #[test]
        fn $name() {
            fn prop(shape: (uint, uint),
                    (low, high): ($ty, $ty))
                    -> TestResult {
                if low >= high {
                    return TestResult::discard();
                }

                let between = Range::new(low, high);
                let mut rng = rand::task_rng();
                let rng = &mut rng;

                let xs = mat::from_fn(shape, |_, _| {
                    let re = between.ind_sample(rng);
                    let im = between.ind_sample(rng);

                    Complex::new(re, im)
                });
                let ys = mat::from_fn(shape, |_, _| {
                    let re = between.ind_sample(rng);
                    let im = between.ind_sample(rng);

                    Complex::new(re, im)
                });
                let mut zs = xs.clone();

                zs.$op_assign(&ys);

                let (nrows, ncols) = shape;
                TestResult::from_bool(
                    xs.shape() == zs.shape() &&
                    range(0, nrows).zip(range(0, ncols)).all(|ref i| {
                        xs.index(i).$op(ys.index(i)).eq(zs.index(i))
                    })
                )
            }

            quickcheck(prop);
        }
    }
}

// AddAssign
macro_rules! add_assign {
    ($name:ident, $ty:ty) => {
        op_assign!($name, $ty, add, add_assign)
    }
}

add_assign!(add_assign_fallback, int)
add_assign!(add_assign_saxpy, f32)
add_assign!(add_assign_daxpy, f64)

macro_rules! add_assign_complex {
    ($name:ident, $ty:ty) => {
        op_assign_complex!($name, $ty, add, add_assign)
    }
}

add_assign_complex!(add_assign_caxpy, f32)
add_assign_complex!(add_assign_zaxpy, f64)

// ArrayNorm2
macro_rules! norm2 {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            fn prop(shape: (uint, uint),
                    (low, high): ($ty, $ty))
                    -> TestResult {
                if low >= high {
                    return TestResult::discard();
                }

                let between = Range::new(low, high);
                let mut rng = rand::task_rng();
                let rng = &mut rng;

                let xs = mat::rand(shape, &between, rng);
                let z = xs.norm2();

                let z_ = xs.iter().zip(xs.iter()).map(|(x, y)| {
                    x.mul(y)
                }).sum().sqrt();

                if z_ == num::zero() || z == num::zero() {
                    return TestResult::discard();
                }

                let diff = z / z_ - num::one();
                let tol = tol();

                TestResult::from_bool(diff <= tol && diff >= -tol)
            }

            quickcheck(prop);
        }
    }
}

norm2!(norm2_snrm, f32)
norm2!(norm2_dnrm, f64)

macro_rules! norm2_complex {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            fn prop(shape: (uint, uint),
                    (low, high): ($ty, $ty))
                    -> TestResult {
                if low >= high {
                    return TestResult::discard();
                }

                let between = Range::new(low, high);
                let mut rng = rand::task_rng();
                let rng = &mut rng;

                let xs = mat::from_fn(shape, |_, _| {
                    let re = between.ind_sample(rng);
                    let im = between.ind_sample(rng);

                    Complex::new(re, im)
                });
                let z = xs.norm2();

                let z_ = xs.iter().map(|x| {
                    x.norm_sqr()
                }).sum().sqrt();

                if z_ == num::zero() || z == num::zero() {
                    return TestResult::discard();
                }

                let diff = z / z_ - num::one();
                let tol = tol();

                TestResult::from_bool(diff <= tol && diff >= -tol)
            }

            quickcheck(prop);
        }
    }
}

norm2_complex!(norm2_scnrm2, f32)
norm2_complex!(norm2_dznrm2, f64)

// ArrayScale
macro_rules! scale {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            fn prop(shape: (uint, uint),
                    (low, high): ($ty, $ty))
                    -> TestResult {
                if low >= high {
                    return TestResult::discard();
                }

                let between = Range::new(low, high);
                let mut rng = rand::task_rng();
                let rng = &mut rng;

                let xs = mat::rand(shape, &between, rng);
                let k = between.ind_sample(rng);
                let mut zs = xs.clone();
                zs.scale(k);

                TestResult::from_bool(xs.iter().zip(zs.iter()).all(|(x, z)| {
                    x.mul(&k).eq(z)
                }))
            }

            quickcheck(prop);
        }
    }
}

scale!(scale_fallback, int)
scale!(scale_sscal, f32)
scale!(scale_dscal, f64)

macro_rules! scale_complex {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            fn prop(shape: (uint, uint),
                    (low, high): ($ty, $ty))
                    -> TestResult {
                if low >= high {
                    return TestResult::discard();
                }

                let between = Range::new(low, high);
                let mut rng = rand::task_rng();
                let rng = &mut rng;

                let xs = mat::from_fn(shape, |_, _| {
                    let re = between.ind_sample(rng);
                    let im = between.ind_sample(rng);

                    Complex::new(re, im)
                });

                let re = between.ind_sample(rng);
                let im = between.ind_sample(rng);
                let k = Complex::new(re, im);

                let mut zs = xs.clone();
                zs.scale(k);

                TestResult::from_bool(xs.iter().zip(zs.iter()).all(|(x, z)| {
                    x.mul(&k).eq(z)
                }))
            }

            quickcheck(prop);
        }
    }
}

scale_complex!(scale_cscal, f32)
scale_complex!(scale_zscal, f64)

// Index
#[test]
fn index() {
    fn prop(shape@(nrows, ncols): (uint, uint),
            index@(row, col): (uint, uint))
            -> TestResult {
        if row >= nrows || col >= ncols {
            return TestResult::discard();
        }

        let xs = mat::from_fn(shape, |i, j| (i, j));
        let i = &index;

        TestResult::from_bool(xs.index(i).eq(i))
    }

    quickcheck(prop);
}

#[test]
#[should_fail]
fn out_of_bounds() {
    fn prop(shape@(nrows, ncols): (uint, uint),
            index@(row, col): (uint, uint))
            -> TestResult {
        if col < ncols || row < nrows {
            return TestResult::discard();
        }

        let xs = mat::from_fn(shape, |i, j| (i, j));
        let i = &index;

        TestResult::from_bool(xs.index(i) == &(0, 0))
    }

    quickcheck(prop);
}

// MulAssign
macro_rules! mul_assign {
    ($name:ident, $ty:ty) => {
        op_assign!($name, $ty, mul, mul_assign)
    }
}

mul_assign!(mul_assign_fallback, int)
mul_assign!(mul_assign_f32x4, f32)
mul_assign!(mul_assign_f64x2, f64)

// SubAssign
macro_rules! sub_assign {
    ($name:ident, $ty:ty) => {
        op_assign!($name, $ty, sub, sub_assign)
    }
}

sub_assign!(sub_assign_fallback, int)
sub_assign!(sub_assign_saxpy, f32)
sub_assign!(sub_assign_daxpy, f64)

macro_rules! sub_assign_complex {
    ($name:ident, $ty:ty) => {
        op_assign_complex!($name, $ty, sub, sub_assign)
    }
}

sub_assign_complex!(sub_assign_caxpy, f32)
sub_assign_complex!(sub_assign_zaxpy, f64)
