use num::Complex;
use std::num;
use std::rand::{Rng, XorShiftRng, mod};

use Mat;

#[allow(non_camel_case_types)]
pub type c64 = Complex<f32>;
#[allow(non_camel_case_types)]
pub type c128 = Complex<f64>;

// NB Sadly, `Complex` does not implements `Rand` in the standard library
trait Rand { fn gen<R>(rng: &mut R) -> Self where R: Rng; }

impl Rand for c128 {
    fn gen<R>(rng: &mut R) -> c128 where R: Rng {
        Complex::new(rng.gen(), rng.gen())
    }
}

impl Rand for c64 {
    fn gen<R>(rng: &mut R) -> c64 where R: Rng {
        Complex::new(rng.gen(), rng.gen())
    }
}

impl Rand for f32 {
    fn gen<R>(rng: &mut R) -> f32 where R: Rng {
        rng.gen()
    }
}
impl Rand for f64 {
    fn gen<R>(rng: &mut R) -> f64 where R: Rng {
        rng.gen()
    }
}

pub fn is_close<T>(x: T, y: T) -> bool where T: Float {
    let tolerance: T = num::cast(1e-5f64).unwrap();

    if x == num::zero() || y == num::zero() {
        (x - y).abs() < tolerance
    } else {
        (x / y - num::one()).abs() < tolerance
    }
}

pub fn mat((nrows, ncols): (uint, uint)) -> Option<Mat<(uint, uint)>> {
    if nrows > 1 && ncols > 1 {
        Some(Mat::from_fn((nrows, ncols), |i| i))
    } else {
        None
    }
}

pub fn rand_mat<T>((nrows, ncols): (uint, uint)) -> Option<Mat<T>> where T: Rand {
    if nrows > 1 && ncols > 1 {
        let mut rng: XorShiftRng = rand::task_rng().gen();

        Some(Mat::from_fn((nrows, ncols), |_| Rand::gen(&mut rng)))
    } else {
        None
    }
}

pub fn size(
    (start_row, start_col): (uint, uint),
    (end_row, end_col): (uint, uint),
) -> (uint, uint) {
    (end_row - start_row, end_col - start_col)
}
