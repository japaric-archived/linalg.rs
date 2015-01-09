#![allow(dead_code)]

use linalg::{ColVec, Mat, RowVec};

pub fn col(length: usize) -> ColVec<usize> {
    ColVec::from_fn(length, |i| i)
}

pub fn mat((nrows, ncols): (usize, usize)) -> Mat<(usize, usize)> {
    Mat::from_fn((nrows, ncols), |i| i).unwrap()
}

pub fn row(length: usize) -> RowVec<usize> {
    RowVec::from_fn(length, |i| i)
}

pub mod rand {
    use std::rand::{Rand, Rng, XorShiftRng, self};

    use linalg::{ColVec, Mat, RowVec};

    pub fn col<T>(length: usize) -> ColVec<T> where T: Rand {
        let ref mut rng: XorShiftRng = rand::thread_rng().gen();

        ColVec::rand(length, rng)
    }

    pub fn mat<T>((nrows, ncols): (usize, usize)) -> Mat<T> where T: Rand {
        let ref mut rng: XorShiftRng = rand::thread_rng().gen();

        Mat::rand((nrows, ncols), rng).unwrap()
    }

    pub fn row<T>(length: usize) -> RowVec<T> where T: Rand {
        let ref mut rng: XorShiftRng = rand::thread_rng().gen();

        RowVec::rand(length, rng)
    }
}

macro_rules! enforce {
    ($($e:expr),+,) => {
        if $(!$e)||+ { return TestResult::discard() }
    }
}

macro_rules! test {
    ($e:expr) => {
        (|&:| Ok::<_, ::linalg::Error>(TestResult::from_bool($e)))().unwrap()
    }
}

macro_rules! validate_diag {
    ($diag:expr, $size:expr) => {{
        let (nrows, ncols) = $size;
        let diag = $diag;

        if diag > 0 {
            let diag = diag as usize;

            enforce! {
                diag < ncols,
            }
        } else {
            let diag = -diag as usize;

            enforce! {
                diag < nrows,
            }
        }
    }}
}

macro_rules! validate_diag_index {
    ($diag:expr, $size:expr, $idx:expr) => {{
        let diag = $diag;
        let (nrows, ncols) = $size;
        let idx = $idx;

        if diag > 0 {
            let diag = diag as usize;

            enforce! {
                diag < ncols,
                idx < ::std::cmp::min(ncols - diag, nrows),
            }
        } else {
            let diag = -diag as usize;

            enforce! {
                diag < nrows,
                idx < ::std::cmp::min(nrows - diag, ncols),
            }
        }
    }}
}
