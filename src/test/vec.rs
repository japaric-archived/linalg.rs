use array::traits::{ArrayDot,ArrayNorm2,ArrayScale,ArrayShape};
use num::complex::Complex;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;
use std::rand;
use std::rand::TaskRng;
use super::NSAMPLES;
// FIXME mozilla/rust#5992 Use std {Add,Mul,Sub}Assign
// FIXME mozilla/rust#6515 Use std Index
use traits::{AddAssign,Index,Iterable,MulAssign,SubAssign};
use vec;

struct RandSizes {
    between: Range<uint>,
    rng: TaskRng,
}

impl Iterator<uint> for RandSizes {
    fn next(&mut self) -> Option<uint> {
        Some(self.between.ind_sample(&mut self.rng))
    }
}

fn rand_sizes() -> RandSizes {
    RandSizes { between: Range::new(100u, 1_000_000), rng: rand::task_rng() }
}

// vec
#[test]
fn from_elem() {
    for n in rand_sizes().take(NSAMPLES) {
        let v = vec::from_elem(n, 0u);

        assert_eq!(v.shape(), (n,));
        assert_eq!(v.unwrap(), Vec::from_elem(n, 0u));
    }
}

#[test]
fn from_fn() {
    for n in rand_sizes().take(NSAMPLES) {
        let v = vec::from_fn(n, |i| i);

        assert_eq!(v.shape(), (n,));
        assert_eq!(v.unwrap(), Vec::from_fn(n, |i| i));
    }
}

#[test]
fn map() {
    for n in rand_sizes().take(NSAMPLES) {
        let mut got = vec::zeros::<f32>(n);
        let expected = vec::ones::<f32>(n);

        got.map(|x| x.cos());

        assert_eq!((n, got), (n, expected));
    }
}

#[test]
fn rand() {
    let between = Range::new(0f64, 1f64);
    let mut rng = rand::task_rng();

    for n in rand_sizes().take(NSAMPLES) {
        let v = vec::rand(n, &between, &mut rng);

        assert_eq!((n, v.all(|&x| x >= 0.0 && x < 1.0)), (n, true));
    }
}

// AddAssign
macro_rules! add_assign {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for n in rand_sizes().take(NSAMPLES) {
                let mut got = vec::from_elem(n, 1 as $ty);
                let v = vec::from_elem(n, 2 as $ty);
                let expected = vec::from_elem(n, 3 as $ty);

                got.add_assign(&v);

                assert_eq!((n, got), (n, expected));
            }
        }
    }
}

add_assign!(add_assign_fallback, int)
add_assign!(add_assign_saxpy, f32)
add_assign!(add_assign_daxpy, f64)

macro_rules! add_assign_complex {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for n in rand_sizes().take(NSAMPLES) {
                let mut got =
                    vec::from_elem(n, Complex::new(1 as $ty, 0 as $ty));
                let v =
                    vec::from_elem(n, Complex::new(0 as $ty, 1 as $ty));
                let expected =
                    vec::from_elem(n, Complex::new(1 as $ty, 1 as $ty));

                got.add_assign(&v);

                assert_eq!((n, got), (n, expected));
            }
        }
    }
}

add_assign_complex!(add_assign_caxpy, f32)
add_assign_complex!(add_assign_zaxpy, f64)

// ArrayDot
macro_rules! dot {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for n in rand_sizes().take(NSAMPLES) {
                let x = vec::ones::<$ty>(n);
                let y = vec::ones::<$ty>(n);
                let got = x.dot(&y);
                let expected = n as $ty;

                assert_eq!((n, got), (n, expected));
            }
        }
    }
}

dot!(dot_fallback, int)
dot!(dot_sdot, f32)
dot!(dot_ddot, f64)

// ArrayNorm2
macro_rules! norm2 {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for n in rand_sizes().take(NSAMPLES) {
                let v = vec::ones::<$ty>(n);
                let expected = (n as $ty).sqrt();
                let got = v.norm2();

                assert_eq!((n, got), (n, expected));
            }
        }
    }
}

norm2!(norm2_snrm, f32)
norm2!(norm2_dnrm, f64)

macro_rules! norm2_complex {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for n in rand_sizes().take(NSAMPLES) {
                let v = vec::from_elem(n, Complex::new(0 as $ty, 1 as $ty));
                let expected = (n as $ty).sqrt();
                let got = v.norm2();

                assert_eq!((n, got), (n, expected));
            }
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
            for n in rand_sizes().take(NSAMPLES) {
                let mut got = vec::ones::<$ty>(n);
                let expected = vec::from_elem(n, 2 as $ty);

                got.scale(2 as $ty);

                assert_eq!((n, got), (n, expected));
            }
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
            for n in rand_sizes().take(NSAMPLES) {
                let mut got =
                    vec::from_elem(n, Complex::new(1 as $ty, 2 as $ty));
                let expected =
                    vec::from_elem(n, Complex::new(-2 as $ty, 1 as $ty));

                got.scale(Complex::new(0 as $ty, 1 as $ty));

                assert_eq!((n, got), (n, expected));
            }
        }
    }
}

scale_complex!(scale_cscal, f32)
scale_complex!(scale_zscal, f64)

// Index
#[test]
fn index() {
    for n in rand_sizes().take(NSAMPLES) {
        let v = vec::from_fn(n, |i| i);

        for i in range(0, n) {
            assert_eq!(v.index(&i), &i);
        }
    }
}

#[test]
#[should_fail]
fn out_of_bounds() {
    let v = vec::zeros::<int>(10);

    v.index(&10);
}

// MulAssign
macro_rules! mul_assign {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for n in rand_sizes().take(NSAMPLES) {
                let mut got = vec::from_elem(n, 2 as $ty);
                let v = vec::from_elem(n, 3 as $ty);
                let expected = vec::from_elem(n, 6 as $ty);

                got.mul_assign(&v);

                assert_eq!((n, got), (n, expected));
            }
        }
    }
}

mul_assign!(mul_assign_fallback, int)
mul_assign!(mul_assign_f32x4, f32)
mul_assign!(mul_assign_f64x2, f64)

// SubAssign
macro_rules! sub_assign {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for n in rand_sizes().take(NSAMPLES) {
                let mut got = vec::from_elem(n, 3 as $ty);
                let v = vec::from_elem(n, 2 as $ty);
                let expected = vec::from_elem(n, 1 as $ty);

                got.sub_assign(&v);

                assert_eq!((n, got), (n, expected));
            }
        }
    }
}

sub_assign!(sub_assign_fallback, int)
sub_assign!(sub_assign_saxpy, f32)
sub_assign!(sub_assign_daxpy, f64)

macro_rules! sub_assign_complex {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for n in rand_sizes().take(NSAMPLES) {
                let mut got =
                    vec::from_elem(n, Complex::new(1 as $ty, 0 as $ty));
                let v =
                    vec::from_elem(n, Complex::new(0 as $ty, 1 as $ty));
                let expected =
                    vec::from_elem(n, Complex::new(1 as $ty, -1 as $ty));

                got.sub_assign(&v);

                assert_eq!((n, got), (n, expected));
            }
        }
    }
}

sub_assign_complex!(sub_assign_caxpy, f32)
sub_assign_complex!(sub_assign_zaxpy, f64)
