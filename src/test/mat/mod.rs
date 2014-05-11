use array::traits::{ArrayNorm2,ArrayScale,ArrayShape};
use std::cmp::min;
use mat::traits::{MatrixCol,MatrixColIterator,MatrixDiag,MatrixRow,
                  MatrixRowIterator};
use mat;
use num::complex::Cmplx;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;
use rand::task_rng;
use super::NSAMPLES;
// FIXME mozilla/rust#5992 Use std {Add,Mul,Sub}Assign
// FIXME mozilla/rust#6515 Use std Index
use traits::{AddAssign,Index,Iterable,MulAssign,SubAssign};

mod view;

fn rand_size() -> (uint, uint) {
    let mut rng = task_rng();
    let between = Range::new(10u, 1_000);

    let nrows = between.ind_sample(&mut rng);
    let ncols = between.ind_sample(&mut rng);

    (nrows, ncols)
}

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! sweep_size {
    ($code:expr) => ({
        for shape@(_nrows, _ncols) in range(0, NSAMPLES).map(|_| rand_size()) {
            $code
        }
    })
}

// mat
#[test]
fn from_elem() {
    sweep_size!({
        let m = mat::from_elem(shape, 0);

        assert_eq!(m.shape(), shape);
        assert_eq!(m.unwrap(), Vec::from_elem(_nrows * _ncols, 0));
    })
}

#[test]
fn from_fn() {
    sweep_size!({
        let m = mat::from_fn(shape, |i, j| i + j);

        assert_eq!(m.shape(), shape);
        assert_eq!(m.unwrap(), Vec::from_fn(_nrows * _ncols, |i| {
            let r = i / _ncols;
            let c = i % _ncols;
            r + c
        }));
    })
}

#[test]
fn map() {
    sweep_size!({
        let mut got = mat::zeros::<f32>(shape);
        let expected = mat::ones::<f32>(shape);

        got.map(|x| x.cos());

        assert_eq!((shape, got), (shape, expected));
    })
}

#[test]
fn rand() {
    let between = Range::new(0.0, 1.0);
    let mut rng = task_rng();

    sweep_size!({
        let v = mat::rand(shape, &between, &mut rng);

        assert_eq!((shape, v.all(|&x| x >= 0.0 && x < 1.0)), (shape, true));
    })
}

// AddAssign
macro_rules! add_assign {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let mut got = mat::from_elem(shape, 1 as $ty);
                let v = mat::from_elem(shape, 2 as $ty);
                let expected = mat::from_elem(shape, 3 as $ty);

                got.add_assign(&v);

                assert_eq!((shape, got), (shape, expected));
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
                    mat::from_elem(shape, Cmplx::new(1 as $ty, 0 as $ty));
                let v =
                    mat::from_elem(shape, Cmplx::new(0 as $ty, 1 as $ty));
                let expected =
                    mat::from_elem(shape, Cmplx::new(1 as $ty, 1 as $ty));

                got.add_assign(&v);

                assert_eq!((shape, got), (shape, expected));
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
                let v = mat::ones::<$ty>(shape);
                let expected = ((_nrows * _ncols) as $ty).sqrt();
                let got = v.norm2();

                assert_eq!((shape, got), (shape, expected));
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
                let v =
                    mat::from_elem(shape, Cmplx::new(0 as $ty, 1 as $ty));
                let expected = ((_nrows * _ncols) as $ty).sqrt();
                let got = v.norm2();

                assert_eq!((shape, got), (shape, expected));
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
                let mut got = mat::ones::<$ty>(shape);
                let expected = mat::from_elem(shape, 2 as $ty);

                got.scale(2 as $ty);

                assert_eq!((shape, got), (shape, expected));
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
                    mat::from_elem(shape, Cmplx::new(1 as $ty, 2 as $ty));
                let expected =
                    mat::from_elem(shape, Cmplx::new(-2 as $ty, 1 as $ty));

                got.scale(Cmplx::new(0 as $ty, 1 as $ty));

                assert_eq!((shape, got), (shape, expected));
            })
        }
    }
}

scale_cmplx!(scale_cscal, f32)
scale_cmplx!(scale_zscal, f64)

// Index
#[test]
#[should_fail]
fn col_out_of_bounds() {
    let v = mat::zeros::<int>((10, 10));

    v.index(&(0, 10));
}

#[test]
fn index() {
    sweep_size!({
        let v = mat::from_fn(shape, |i, j| i + j);

        for i in range(0, _nrows) {
            for j in range(0, _ncols) {
                let got = *v.index(&(i, j));
                let expected = i + j;

                assert_eq!((shape, got), (shape, expected));
            }
        }
    })
}

#[test]
#[should_fail]
fn row_out_of_bounds() {
    let v = mat::zeros::<int>((10, 10));

    v.index(&(10, 0));
}

// MatrixCol
#[test]
fn col() {
    sweep_size!({
        let m = mat::from_fn(shape, |i, j| i - j);

        for j in range(0, _ncols) {
            let col = m.col(j);

            for i in range(0, _nrows) {
                let got = *col.index(&i);
                let expected = i - j;

                assert_eq!((shape, got), (shape, expected));
            }
        }
    })
}

#[test]
fn iterable_col() {
    sweep_size!({
        let m = mat::from_fn(shape, |i, j| i - j);

        for j in range(0, _ncols) {
            let col = m.col(j);
            let got = col.iter().map(|&x| x).collect();
            let expected = Vec::from_fn(_nrows, |i| i - j);

            assert_eq!((shape, got), (shape, expected));
        }
    })
}

// MatrixColIterator
#[test]
fn cols() {
    sweep_size!({
        let m = mat::from_fn(shape, |i, j| i - j);

        for (j, col) in m.cols().enumerate() {
            for i in range(0, _nrows) {
                let got = *col.index(&i);
                let expected = i - j;

                assert_eq!((shape, got), (shape, expected));
            }
        }
    })
}

// MatrixDiag
#[test]
fn diag() {
    sweep_size!({
        let m = mat::from_fn(shape, |i, j| j as int - i as int);

        for d in range(-(_nrows as int) + 1, _ncols as int) {
            let got = m.diag(d).iter().map(|&x| x).collect();
            let expected = if d > 0 {
                Vec::from_elem(min(_nrows, _ncols - d as uint), d)
            } else {
                Vec::from_elem(min(_nrows + d as uint, _ncols), d)
            };

            assert_eq!((shape, d, got), (shape, d, expected))
        }
    })
}

#[test]
#[should_fail]
fn diag_col_out_of_bounds() {
    let m = mat::ones::<int>((4, 3));

    m.diag(3);
}

#[test]
#[should_fail]
fn diag_row_out_of_bounds() {
    let m = mat::ones::<int>((4, 3));

    m.diag(-4);
}

// MatrixRow
#[test]
fn iterable_row() {
    sweep_size!({
        let m = mat::from_fn(shape, |i, j| i - j);

        for i in range(0, _nrows) {
            let row = m.row(i);
            let got = row.iter().map(|&x| x).collect();
            let expected = Vec::from_fn(_ncols, |j| i - j);

            assert_eq!((shape, got), (shape, expected));
        }
    })
}

#[test]
fn row() {
    sweep_size!({
        let m = mat::from_fn(shape, |i, j| i - j);

        for i in range(0, _nrows) {
            let row = m.row(i);

            for j in range(0, _ncols) {
                let got = *row.index(&j);
                let expected = i - j;

                assert_eq!((shape, got), (shape, expected));
            }
        }
    })
}

// MatrixRowIterator
#[test]
fn rows() {
    sweep_size!({
        let m = mat::from_fn(shape, |i, j| i - j);

        for (i, row) in m.rows().enumerate() {
            for j in range(0, _ncols) {
                let got = *row.index(&j);
                let expected = i - j;

                assert_eq!((shape, got), (shape, expected));
            }
        }
    })
}

// MulAssign
macro_rules! mul_assign {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            sweep_size!({
                let mut got = mat::from_elem(shape, 2 as $ty);
                let v = mat::from_elem(shape, 3 as $ty);
                let expected = mat::from_elem(shape, 6 as $ty);

                got.mul_assign(&v);

                assert_eq!((shape, got), (shape, expected));
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
                let mut got = mat::from_elem(shape, 3 as $ty);
                let v = mat::from_elem(shape, 2 as $ty);
                let expected = mat::from_elem(shape, 1 as $ty);

                got.sub_assign(&v);

                assert_eq!((shape, got), (shape, expected));
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
                    mat::from_elem(shape, Cmplx::new(1 as $ty, 0 as $ty));
                let v =
                    mat::from_elem(shape, Cmplx::new(0 as $ty, 1 as $ty));
                let expected =
                    mat::from_elem(shape, Cmplx::new(1 as $ty, -1 as $ty));

                got.sub_assign(&v);

                assert_eq!((shape, got), (shape, expected));
            })
        }
    }
}

sub_assign_cmplx!(sub_assign_caxpy, f32)
sub_assign_cmplx!(sub_assign_zaxpy, f64)
