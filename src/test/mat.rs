use array::traits::ArrayScale;
use mat;
use num::complex::Cmplx;
use rand::distributions::range::Range;
use rand::task_rng;
use traits::{AddAssign,Iterable,MulAssign,SubAssign};

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! sweep_size {
    ($code:expr) => ({
        for &n in [10, 100, 1_000].iter() {
            $code
        }
    })
}

// mat
#[test]
fn from_elem() {
    sweep_size!({
        let m = mat::from_elem((n, n), 0);

        assert_eq!(m.shape(), (n, n));
        assert_eq!(m.unwrap(), Vec::from_elem(n * n, 0));
    })
}

#[test]
fn from_fn() {
    sweep_size!({
        let m = mat::from_fn((n, n), |i, j| i + j);

        assert_eq!(m.shape(), (n, n));
        assert_eq!(m.unwrap(), Vec::from_fn(n * n, |i| {
            let r = i / n;
            let c = i % n;
            r + c
        }));
    })
}

#[test]
fn rand() {
    let between = Range::new(0.0, 1.0);
    let mut rng = task_rng();

    sweep_size!({
        let v = mat::rand((n, n), &between, &mut rng);

        assert_eq!((n, v.all(|&x| x >= 0.0 && x < 1.0)), (n, true));
    })
}

// AddAssign
macro_rules! add_assign {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let mut got = mat::from_elem((n, n), 1 as $ty);
                let v = mat::from_elem((n, n), 2 as $ty);
                let expected = mat::from_elem((n, n), 3 as $ty);

                got.add_assign(&v);

                assert_eq!((n, got), (n, expected));
            })
        }
    }
}

add_assign!(add_assign_fallback, int)
add_assign!(add_assign_saxpy, f32)
add_assign!(add_assign_daxpy, f64)

macro_rules! add_assign_cmplx {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let mut got =
                    mat::from_elem((n, n), Cmplx::new(1 as $ty, 0 as $ty));
                let v =
                    mat::from_elem((n, n), Cmplx::new(0 as $ty, 1 as $ty));
                let expected =
                    mat::from_elem((n, n), Cmplx::new(1 as $ty, 1 as $ty));

                got.add_assign(&v);

                assert_eq!((n, got), (n, expected));
            })
        }
    }
}

add_assign_cmplx!(add_assign_caxpy, f32)
add_assign_cmplx!(add_assign_zaxpy, f64)

// ArrayScale
macro_rules! scale {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let mut got = mat::ones::<$ty>((n, n));
                let expected = mat::from_elem((n, n), 2 as $ty);

                got.scale(2 as $ty);

                assert_eq!((n, got), (n, expected));
            })
        }
    }
}

scale!(scale_fallback, int)
scale!(scale_sscal, f32)
scale!(scale_dscal, f64)

macro_rules! scale_cmplx {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let mut got =
                    mat::from_elem((n, n), Cmplx::new(1 as $ty, 2 as $ty));
                let expected =
                    mat::from_elem((n, n), Cmplx::new(-2 as $ty, 1 as $ty));

                got.scale(Cmplx::new(0 as $ty, 1 as $ty));

                assert_eq!((n, got), (n, expected));
            })
        }
    }
}

scale_cmplx!(scale_cscal, f32)
scale_cmplx!(scale_zscal, f64)

// MulAssign
macro_rules! mul_assign {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let mut got = mat::from_elem((n, n), 2 as $ty);
                let v = mat::from_elem((n, n), 3 as $ty);
                let expected = mat::from_elem((n, n), 6 as $ty);

                got.mul_assign(&v);

                assert_eq!((n, got), (n, expected));
            })
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
            sweep_size!({
                let mut got = mat::from_elem((n, n), 3 as $ty);
                let v = mat::from_elem((n, n), 2 as $ty);
                let expected = mat::from_elem((n, n), 1 as $ty);

                got.sub_assign(&v);

                assert_eq!((n, got), (n, expected));
            })
        }
    }
}

sub_assign!(sub_assign_fallback, int)
sub_assign!(sub_assign_saxpy, f32)
sub_assign!(sub_assign_daxpy, f64)

macro_rules! sub_assign_cmplx {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let mut got =
                    mat::from_elem((n, n), Cmplx::new(1 as $ty, 0 as $ty));
                let v =
                    mat::from_elem((n, n), Cmplx::new(0 as $ty, 1 as $ty));
                let expected =
                    mat::from_elem((n, n), Cmplx::new(1 as $ty, -1 as $ty));

                got.sub_assign(&v);

                assert_eq!((n, got), (n, expected));
            })
        }
    }
}

sub_assign_cmplx!(sub_assign_caxpy, f32)
sub_assign_cmplx!(sub_assign_zaxpy, f64)
