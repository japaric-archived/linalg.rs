use std::num;
use std::rand::{Rand, Rng, XorShiftRng, mod};

use Mat;

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

        Some(Mat::rand((nrows, ncols), &mut rng))
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
