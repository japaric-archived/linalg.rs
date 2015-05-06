#![allow(dead_code)]

use linalg::prelude::*;

pub mod rand;

macro_rules! enforce {
    ($($e:expr),+,) => {
        if !($($e &&)+ true) {
            return TestResult::discard()
        };
    }
}

macro_rules! test {
    ($e:expr) => {
        if $e {
            TestResult::passed()
        } else {
            return TestResult::failed()
        }
    };
}

macro_rules! test_approx_eq {
    ($lhs:expr, $rhs:expr) => {{
        let ref lhs = $lhs;
        let ref rhs = $rhs;

        test!{
            ::approx::eq(lhs, rhs, ::approx::Abs::tol(1e-3)) ||
            ::approx::eq(lhs, rhs, ::approx::Rel::tol(1e-3))
        }
    }};
}

macro_rules! test_eq {
    ($lhs:expr, $rhs:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;

        test!(lhs == rhs)
    }};
}

macro_rules! validate_diag_index {
    ($size:expr, $diag:expr, $i:expr) => {{
        use cast::From;

        let (nrows, ncols) = $size;
        let diag = $diag;
        let i = $i;

        if diag > 0 {
            let diag = u32::from(diag).unwrap();

            enforce! {
                diag < ncols,
            }

            let n = ::std::cmp::min(ncols - diag, nrows);

            enforce! {
                i < n,
            }

            n
        } else {
            let diag = u32::from(-diag).unwrap();

            enforce! {
                diag < nrows,
            }

            let n = ::std::cmp::min(nrows - diag, ncols);

            enforce! {
                i < n,
            }

            n
        }
    }};
}

pub fn col(len: u32) -> ColVec<u32> {
    (0..len).map(|i| i).collect()
}

pub fn mat(size: (u32, u32)) -> Mat<(u32, u32)> {
    Mat::from_fn(size, |i| i)
}

pub fn row(len: u32) -> RowVec<u32> {
    (0..len).map(|i| i).collect()
}
