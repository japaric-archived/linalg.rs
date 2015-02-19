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

                // Test that `sub_assign(&Trans<Mat>)` is correct for `Mat`
                #[quickcheck]
                fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
                    enforce! {
                        row < nrows,
                        col < ncols,
                    }

                    let idx = (row, col);

                    test!({
                        let mut result = setup::rand::mat::<$ty>((nrows, ncols));
                        let &lhs = try!(result.at(idx));

                        let rhs = setup::rand::mat::<$ty>((ncols, nrows)).t();

                        result.sub_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs - rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `sub_assign(&Trans<View>)` is correct for `Mat`
                #[quickcheck]
                fn view(
                    start: (usize, usize),
                    (nrows, ncols): (usize, usize),
                    (row, col): (usize, usize),
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        col < ncols,
                    }

                    let size = (start.0 + ncols, start.1 + nrows);
                    let idx = (row, col);

                    test!({
                        let mut result = setup::rand::mat::<$ty>((nrows, ncols));
                        let &lhs = try!(result.at(idx));

                        let m = setup::rand::mat::<$ty>(size);
                        let rhs = try!(m.slice_from(start)).t();

                        result.sub_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs - rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `sub_assign(&Trans<MutView>)` is correct for `Mat`
                #[quickcheck]
                fn view_mut(
                    start: (usize, usize),
                    (nrows, ncols): (usize, usize),
                    (row, col): (usize, usize),
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        col < ncols,
                    }

                    let size = (start.0 + ncols, start.1 + nrows);
                    let idx = (row, col);

                    test!({
                        let mut result = setup::rand::mat::<$ty>((nrows, ncols));
                        let &lhs = try!(result.at(idx));

                        let mut m = setup::rand::mat::<$ty>(size);
                        let rhs = try!(m.slice_from_mut(start)).t();

                        result.sub_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs - rhs, *try!(result.at(idx)))
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

            // Test that `sub_assign(&Mat)` is correct for `Mat`
            #[quickcheck]
            fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
                enforce! {
                    row < nrows,
                    col < ncols,
                }

                let size = (nrows, ncols);
                let idx = (row, col);

                test!({
                    let mut result = setup::rand::mat::<$ty>(size);
                    let &lhs = try!(result.at(idx));

                    let rhs = setup::rand::mat::<$ty>(size);

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    approx_eq!(lhs - rhs, *try!(result.at(idx)))
                })
            }

            // Test that `sub_assign(&T)` is correct for `Mat`
            #[quickcheck]
            fn scalar((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
                enforce! {
                    row < nrows,
                    col < ncols,
                }

                let size = (nrows, ncols);
                let idx = (row, col);

                test!({
                    let mut result = setup::rand::mat::<$ty>(size);
                    let &lhs = try!(result.at(idx));

                    let rhs: $ty = ::rand::random();

                    result.sub_assign(&rhs);

                    approx_eq!(lhs - rhs, *try!(result.at(idx)))
                })
            }

            // Test that `sub_assign(&View)` is correct for `Mat`
            #[quickcheck]
            fn view(
                start: (usize, usize),
                (nrows, ncols): (usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < nrows,
                    col < ncols,
                }

                let size = (start.0 + nrows, start.1 + ncols);
                let idx = (row, col);

                test!({
                    let mut result = setup::rand::mat::<$ty>((nrows, ncols));
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>(size);
                    let rhs = try!(m.slice_from(start));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    approx_eq!(lhs - rhs, *try!(result.at(idx)))
                })
            }

            // Test that `sub_assign(&MutView)` is correct for `Mat`
            #[quickcheck]
            fn view_mut(
                start: (usize, usize),
                (nrows, ncols): (usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < nrows,
                    col < ncols,
                }

                let size = (start.0 + nrows, start.1 + ncols);
                let idx = (row, col);

                test!({
                    let mut result = setup::rand::mat::<$ty>((nrows, ncols));
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>(size);
                    let rhs = try!(m.slice_from_mut(start));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    approx_eq!(lhs - rhs, *try!(result.at(idx)))
                })
            }
        }
    }
}

blas!(f32);
blas!(f64);
blas!(c64);
blas!(c128);
