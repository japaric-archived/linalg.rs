#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate approx;
extern crate linalg;
extern crate quickcheck;
extern crate rand;

#[macro_use]
mod setup;

mod trans {
    macro_rules! blas {
        ($ty:ident) => {
            mod $ty {
                use linalg::prelude::*;
                use quickcheck::TestResult;

                use setup;

                // Test that `add_assign(&Trans<Mat>)` is correct for `Trans<MutView>`
                #[quickcheck]
                fn mat(
                    start: (usize, usize),
                    (nrows, ncols): (usize, usize),
                    (row, col): (usize, usize),
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        col < ncols,
                    }

                    let idx = (row, col);
                    let size = (start.0 + ncols, start.1 + nrows);
                    test!({
                        let mut m = setup::rand::mat::<$ty>(size);
                        let mut result = try!(m.slice_mut(start..)).t();
                        let &lhs = try!(result.at(idx));

                        let rhs = setup::rand::mat::<$ty>((ncols, nrows)).t();

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&Trans<View>)` is correct for `Trans<MutView>`
                #[quickcheck]
                fn view(
                    (lhs_start, rhs_start): ((usize, usize), (usize, usize)),
                    (nrows, ncols): (usize, usize),
                    (row, col): (usize, usize),
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        col < ncols,
                    }

                    let idx = (row, col);
                    let lhs_size = (lhs_start.0 + ncols, lhs_start.1 + nrows);
                    let rhs_size = (rhs_start.0 + ncols, rhs_start.1 + nrows);
                    test!({
                        let mut m = setup::rand::mat::<$ty>(lhs_size);
                        let mut result = try!(m.slice_mut(lhs_start..)).t();
                        let &lhs = try!(result.at(idx));

                        let m = setup::rand::mat::<$ty>(rhs_size);
                        let rhs = try!(m.slice(rhs_start..)).t();

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&Trans<MutView>)` is correct for `Trans<MutView>`
                #[quickcheck]
                fn view_mut(
                    (lhs_start, rhs_start): ((usize, usize), (usize, usize)),
                    (nrows, ncols): (usize, usize),
                    (row, col): (usize, usize),
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        col < ncols,
                    }

                    let idx = (row, col);
                    let lhs_size = (lhs_start.0 + ncols, lhs_start.1 + nrows);
                    let rhs_size = (rhs_start.0 + ncols, rhs_start.1 + nrows);
                    test!({
                        let mut m = setup::rand::mat::<$ty>(lhs_size);
                        let mut result = try!(m.slice_mut(lhs_start..)).t();
                        let &lhs = try!(result.at(idx));

                        let mut m = setup::rand::mat::<$ty>(rhs_size);
                        let rhs = try!(m.slice_mut(rhs_start..)).t();

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }
            }
        }
    }

    blas!(f32);
    blas!(f64);
    blas!(c64);
    blas!(c128);
}

macro_rules! blas {
    ($ty:ident) => {
        mod $ty {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `add_assign(&Mat)` is correct for `Trans<MutView>`
            #[quickcheck]
            fn mat(
                start: (usize, usize),
                (nrows, ncols): (usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < nrows,
                    col < ncols,
                }

                let idx = (row, col);
                let size = (start.0 + ncols, start.1 + nrows);
                test!({
                    let mut m = setup::rand::mat::<$ty>(size);
                    let mut result = try!(m.slice_mut(start..)).t();
                    let &lhs = try!(result.at(idx));

                    let rhs = setup::rand::mat::<$ty>((nrows, ncols));

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    approx_eq!(lhs + rhs, *try!(result.at(idx)))
                })
            }

            // Test that `add_assign(&T)` is correct for `Trans<MutView>`
            #[quickcheck]
            fn scalar(
                start: (usize, usize),
                (nrows, ncols): (usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < nrows,
                    col < ncols,
                }

                let idx = (row, col);
                let size = (start.0 + ncols, start.1 + nrows);
                test!({
                    let mut m = setup::rand::mat::<$ty>(size);
                    let mut result = try!(m.slice_mut(start..)).t();
                    let &lhs = try!(result.at(idx));

                    let rhs: $ty = ::rand::random();

                    result.add_assign(&rhs);

                    approx_eq!(lhs + rhs, *try!(result.at(idx)))
                })
            }

            // Test that `add_assign(&View)` is correct for `Trans<MutView>`
            #[quickcheck]
            fn view(
                (lhs_start, rhs_start): ((usize, usize), (usize, usize)),
                (nrows, ncols): (usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < nrows,
                    col < ncols,
                }

                let idx = (row, col);
                let lhs_size = (lhs_start.0 + ncols, lhs_start.1 + nrows);
                let rhs_size = (rhs_start.0 + nrows, rhs_start.1 + ncols);
                test!({
                    let mut m = setup::rand::mat::<$ty>(lhs_size);
                    let mut result = try!(m.slice_mut(lhs_start..)).t();
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>(rhs_size);
                    let rhs = try!(m.slice(rhs_start..));

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    approx_eq!(lhs + rhs, *try!(result.at(idx)))
                })
            }

            // Test that `add_assign(&MutView)` is correct for `Trans<MutView>`
            #[quickcheck]
            fn view_mut(
                (lhs_start, rhs_start): ((usize, usize), (usize, usize)),
                (nrows, ncols): (usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < nrows,
                    col < ncols,
                }

                let idx = (row, col);
                let lhs_size = (lhs_start.0 + ncols, lhs_start.1 + nrows);
                let rhs_size = (rhs_start.0 + nrows, rhs_start.1 + ncols);
                test!({
                    let mut m = setup::rand::mat::<$ty>(lhs_size);
                    let mut result = try!(m.slice_mut(lhs_start..)).t();
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>(rhs_size);
                    let rhs = try!(m.slice_mut(rhs_start..));

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    approx_eq!(lhs + rhs, *try!(result.at(idx)))
                })
            }
        }
    }
}

blas!(f32);
blas!(f64);
blas!(c64);
// FIXME(#46)
//blas!(c128);
