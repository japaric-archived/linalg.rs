use array::traits::{ArrayNorm2,ArrayScale};
use num::complex::Cmplx;
use rand::distributions::range::Range;
use rand::task_rng;
use traits::{AddAssign,Iterable,MulAssign,SubAssign};
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! sweep_size {
    ($code:expr) => ({
        for &n in [1_000, 100_000, 1_000_000].iter() {
            $code
        }
    })
}

// vec
#[test]
fn from_elem() {
    sweep_size!({
        let v = vec::from_elem(n, 0);

        assert_eq!(v.shape(), (n,));
        assert_eq!(v.unwrap(), Vec::from_elem(n, 0));
    })
}

#[test]
fn from_fn() {
    sweep_size!({
        let v = vec::from_fn(n, |i| i);

        assert_eq!(v.shape(), (n,));
        assert_eq!(v.unwrap(), Vec::from_fn(n, |i| i));
    })
}

#[test]
fn rand() {
    let between = Range::new(0.0, 1.0);
    let mut rng = task_rng();

    sweep_size!({
        let v = vec::rand(n, &between, &mut rng);

        assert_eq!((n, v.all(|&x| x >= 0.0 && x < 1.0)), (n, true));
    })
}

// AddAssign
macro_rules! add_assign {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let mut got = vec::from_elem(n, 1 as $ty);
                let v = vec::from_elem(n, 2 as $ty);
                let expected = vec::from_elem(n, 3 as $ty);

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
                    vec::from_elem(n, Cmplx::new(1 as $ty, 0 as $ty));
                let v =
                    vec::from_elem(n, Cmplx::new(0 as $ty, 1 as $ty));
                let expected =
                    vec::from_elem(n, Cmplx::new(1 as $ty, 1 as $ty));

                got.add_assign(&v);

                assert_eq!((n, got), (n, expected));
            })
        }
    }
}

add_assign_cmplx!(add_assign_caxpy, f32)
add_assign_cmplx!(add_assign_zaxpy, f64)

// ArrayNorm2
macro_rules! norm2 {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let v = vec::ones::<$ty>(n);
                let expected = (n as $ty).sqrt();
                let got = v.norm2();

                assert_eq!((n, got), (n, expected));
            })
        }
    }
}

norm2!(norm2_snrm, f32)
norm2!(norm2_dnrm, f64)

macro_rules! norm2_cmplx {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let v = vec::from_elem(n, Cmplx::new(0 as $ty, 1 as $ty));
                let expected = (n as $ty).sqrt();
                let got = v.norm2();

                assert_eq!((n, got), (n, expected));
            })
        }
    }
}

norm2_cmplx!(norm2_scnrm2, f32)
norm2_cmplx!(norm2_dznrm2, f64)

// ArrayScale
macro_rules! scale {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let mut got = vec::ones::<$ty>(n);
                let expected = vec::from_elem(n, 2 as $ty);

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
                    vec::from_elem(n, Cmplx::new(1 as $ty, 2 as $ty));
                let expected =
                    vec::from_elem(n, Cmplx::new(-2 as $ty, 1 as $ty));

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
                let mut got = vec::from_elem(n, 2 as $ty);
                let v = vec::from_elem(n, 3 as $ty);
                let expected = vec::from_elem(n, 6 as $ty);

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
                let mut got = vec::from_elem(n, 3 as $ty);
                let v = vec::from_elem(n, 2 as $ty);
                let expected = vec::from_elem(n, 1 as $ty);

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
                    vec::from_elem(n, Cmplx::new(1 as $ty, 0 as $ty));
                let v =
                    vec::from_elem(n, Cmplx::new(0 as $ty, 1 as $ty));
                let expected =
                    vec::from_elem(n, Cmplx::new(1 as $ty, -1 as $ty));

                got.sub_assign(&v);

                assert_eq!((n, got), (n, expected));
            })
        }
    }
}

sub_assign_cmplx!(sub_assign_caxpy, f32)
sub_assign_cmplx!(sub_assign_zaxpy, f64)
