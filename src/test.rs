use num::Complex;
use std::num;
use std::rand::{Rng, XorShiftRng, mod};

use Mat;

#[allow(non_camel_case_types)]
pub type c64 = Complex<f32>;
#[allow(non_camel_case_types)]
pub type c128 = Complex<f64>;

// NB Sadly, `Complex` does not implements `Rand` in the standard library
trait Rand { fn gen<R: Rng>(rng: &mut R) -> Self; }

impl Rand for c128 { fn gen<R: Rng>(rng: &mut R) -> c128 { Complex::new(rng.gen(), rng.gen()) } }
impl Rand for c64 { fn gen<R: Rng>(rng: &mut R) -> c64 { Complex::new(rng.gen(), rng.gen()) } }
impl Rand for f32 { fn gen<R: Rng>(rng: &mut R) -> f32 { rng.gen() } }
impl Rand for f64 { fn gen<R: Rng>(rng: &mut R) -> f64 { rng.gen() } }

pub fn is_close<T: Float>(x: T, y: T) -> bool {
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

pub fn rand_mat<T: Rand>((nrows, ncols): (uint, uint)) -> Option<Mat<T>> {
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
