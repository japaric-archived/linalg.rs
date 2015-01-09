#![allow(unstable)]
#![feature(plugin)]

extern crate linalg;
extern crate quickcheck;
#[plugin]
extern crate quickcheck_macros;

#[macro_use]
mod setup;

mod trans {
    macro_rules! blas {
        ($($ty:ident),+) => {$(mod $ty {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `add_assign(&Trans<Mat>)` is correct for `Trans<Mat>`
            #[quickcheck]
            fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
                enforce! {
                    row < nrows,
                    col < ncols,
                }

                let idx = (row, col);

                test!({
                    let mut result = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let &lhs = try!(result.at(idx));

                    let rhs = setup::rand::mat::<$ty>((ncols, nrows)).t();

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(Trans<View>)` is correct for `Trans<Mat>`
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

                let idx = (row, col);

                let size = (start.0 + ncols, start.1 + nrows);
                test!({
                    let mut result = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>(size);
                    let rhs = try!(m.slice_from(start)).t();

                    result.add_assign(rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(&Trans<MutView>)` is correct for `Trans<Mat>`
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

                let idx = (row, col);

                let size = (start.0 + ncols, start.1 + nrows);
                test!({
                    let mut result = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>(size);
                    let rhs = try!(m.slice_from_mut(start)).t();

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }})+
        }
    }

    blas!(f32, f64, c64, c128);
}

macro_rules! blas {
    ($($ty:ident),+) => {$(mod $ty {
        use linalg::prelude::*;
        use quickcheck::TestResult;

        use setup;

        // Test that `add_assign(&Mat)` is correct for `Trans<Mat>`
        #[quickcheck]
        fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
            enforce! {
                row < nrows,
                col < ncols,
            }

            let idx = (row, col);

            test!({
                let mut result = setup::rand::mat::<$ty>((ncols, nrows)).t();
                let &lhs = try!(result.at(idx));

                let rhs = setup::rand::mat::<$ty>((nrows, ncols));

                result.add_assign(&rhs);

                let &rhs = try!(rhs.at(idx));

                lhs + rhs == *try!(result.at(idx))
            })
        }

        // Test that `add_assign(T)` is correct for `Trans<Mat>`
        #[quickcheck]
        fn scalar((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
            enforce! {
                row < nrows,
                col < ncols,
            }

            let idx = (row, col);

            test!({
                let mut result = setup::rand::mat::<$ty>((ncols, nrows)).t();
                let &lhs = try!(result.at(idx));

                let rhs: $ty = ::std::rand::random();

                result.add_assign(rhs);

                lhs + rhs == *try!(result.at(idx))
            })
        }

        // Test that `add_assign(View)` is correct for `Trans<Mat>`
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

            let idx = (row, col);

            let size = (start.0 + nrows, start.1 + ncols);
            test!({
                let mut result = setup::rand::mat::<$ty>((ncols, nrows)).t();
                let &lhs = try!(result.at(idx));

                let m = setup::rand::mat::<$ty>(size);
                let rhs = try!(m.slice_from(start));

                result.add_assign(rhs);

                let &rhs = try!(rhs.at(idx));

                lhs + rhs == *try!(result.at(idx))
            })
        }

        // Test that `add_assign(&MutView)` is correct for `Trans<Mat>`
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

            let idx = (row, col);

            let size = (start.0 + nrows, start.1 + ncols);
            test!({
                let mut result = setup::rand::mat::<$ty>((ncols, nrows)).t();
                let &lhs = try!(result.at(idx));

                let mut m = setup::rand::mat::<$ty>(size);
                let rhs = try!(m.slice_from_mut(start));

                result.add_assign(&rhs);

                let &rhs = try!(rhs.at(idx));

                lhs + rhs == *try!(result.at(idx))
            })
        }})+
    }
}

blas!(f32, f64, c64, c128);
