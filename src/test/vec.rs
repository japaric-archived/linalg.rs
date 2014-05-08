use num::complex::Cmplx;
use rand::distributions::range::Range;
use rand::task_rng;
use traits::{AddAssign,Iterable,SubAssign};
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! sweep_size {
    ($code:expr) => ({
        for &n in [1_000, 100_000, 1_000_000].iter() {
            $code
        }
    })
}

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
