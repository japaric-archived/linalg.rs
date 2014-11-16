#![allow(dead_code)]
#![macro_escape]

use linalg::{Col, Mat, Row};

pub fn col(length: uint) -> Col<Box<[uint]>> {
    Col::from_fn(length, |i| i)
}

pub fn mat((nrows, ncols): (uint, uint)) -> Mat<(uint, uint)> {
    Mat::from_fn((nrows, ncols), |i| i).unwrap()
}

pub fn row(length: uint) -> Row<Box<[uint]>> {
    Row::from_fn(length, |i| i)
}

pub mod rand {
    use std::rand::{Rand, Rng, XorShiftRng, mod};

    use linalg::{Col, Mat, Row};

    pub fn col<T>(length: uint) -> Col<Box<[T]>> where T: Rand {
        let ref mut rng: XorShiftRng = rand::task_rng().gen();

        Col::rand(length, rng)
    }

    pub fn mat<T>((nrows, ncols): (uint, uint)) -> Mat<T> where T: Rand {
        let ref mut rng: XorShiftRng = rand::task_rng().gen();

        Mat::rand((nrows, ncols), rng).unwrap()
    }

    pub fn row<T>(length: uint) -> Row<Box<[T]>> where T: Rand {
        let ref mut rng: XorShiftRng = rand::task_rng().gen();

        Row::rand(length, rng)
    }
}

macro_rules! enforce {
    ($($e:expr),+,) => {
        if $(!$e)||+ { return TestResult::discard() }
    }
}

macro_rules! test {
    ($e:expr) => {
        (|| Ok::<_, ::linalg::Error>(TestResult::from_bool($e)))().unwrap()
    }
}

macro_rules! validate_diag {
    ($diag:expr, $size:expr) => {{
        let (nrows, ncols) = $size;
        let diag = $diag;

        if diag > 0 {
            let diag = diag as uint;

            enforce!{
                diag < ncols,
            }
        } else {
            let diag = -diag as uint;

            enforce!{
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
            let diag = diag as uint;

            enforce!{
                diag < ncols,
                idx < ::std::cmp::min(ncols - diag, nrows),
            }
        } else {
            let diag = -diag as uint;

            enforce!{
                diag < nrows,
                idx < ::std::cmp::min(nrows - diag, ncols),
            }
        }
    }}
}
