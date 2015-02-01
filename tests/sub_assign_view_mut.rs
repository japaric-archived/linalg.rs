#![feature(plugin)]
#![feature(rand)]

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

            // Test that `sub_assign(&Trans<Mat>)` is correct for `MutView`
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
                let size = (start.0 + nrows, start.1 + ncols);
                test!({
                    let mut m = setup::rand::mat::<$ty>(size);
                    let mut result = try!(m.slice_from_mut(start));
                    let &lhs = try!(result.at(idx));

                    let rhs = setup::rand::mat::<$ty>((ncols, nrows)).t();

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Trans<View>)` is correct for `MutView`
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
                let lhs_size = (lhs_start.0 + nrows, lhs_start.1 + ncols);
                let rhs_size = (rhs_start.0 + ncols, rhs_start.1 + nrows);
                test!({
                    let mut m = setup::rand::mat::<$ty>(lhs_size);
                    let mut result = try!(m.slice_from_mut(lhs_start));
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>(rhs_size);
                    let rhs = try!(m.slice_from(rhs_start)).t();

                    result.sub_assign(rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(&Trans<MutView>)` is correct for `MutView`
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
                let lhs_size = (lhs_start.0 + nrows, lhs_start.1 + ncols);
                let rhs_size = (rhs_start.0 + ncols, rhs_start.1 + nrows);
                test!({
                    let mut m = setup::rand::mat::<$ty>(lhs_size);
                    let mut result = try!(m.slice_from_mut(lhs_start));
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>(rhs_size);
                    let rhs = try!(m.slice_from_mut(rhs_start)).t();

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
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

        // Test that `sub_assign(&Mat)` is correct for `MutView`
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
            let size = (start.0 + nrows, start.1 + ncols);
            test!({
                let mut m = setup::rand::mat::<$ty>(size);
                let mut result = try!(m.slice_from_mut(start));
                let &lhs = try!(result.at(idx));

                let rhs = setup::rand::mat::<$ty>((nrows, ncols));

                result.sub_assign(&rhs);

                let &rhs = try!(rhs.at(idx));

                lhs - rhs == *try!(result.at(idx))
            })
        }

        // Test that `sub_assign(T)` is correct for `MutView`
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
            let size = (start.0 + nrows, start.1 + ncols);
            test!({
                let mut m = setup::rand::mat::<$ty>(size);
                let mut result = try!(m.slice_from_mut(start));
                let &lhs = try!(result.at(idx));

                let rhs: $ty = ::std::rand::random();

                result.sub_assign(rhs);

                lhs - rhs == *try!(result.at(idx))
            })
        }

        // Test that `sub_assign(View)` is correct for `MutView`
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
            let lhs_size = (lhs_start.0 + nrows, lhs_start.1 + ncols);
            let rhs_size = (rhs_start.0 + nrows, rhs_start.1 + ncols);
            test!({
                let mut m = setup::rand::mat::<$ty>(lhs_size);
                let mut result = try!(m.slice_from_mut(lhs_start));
                let &lhs = try!(result.at(idx));

                let m = setup::rand::mat::<$ty>(rhs_size);
                let rhs = try!(m.slice_from(rhs_start));

                result.sub_assign(rhs);

                let &rhs = try!(rhs.at(idx));

                lhs - rhs == *try!(result.at(idx))
            })
        }

        // Test that `sub_assign(&MutView)` is correct for `MutView`
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
            let lhs_size = (lhs_start.0 + nrows, lhs_start.1 + ncols);
            let rhs_size = (rhs_start.0 + nrows, rhs_start.1 + ncols);
            test!({
                let mut m = setup::rand::mat::<$ty>(lhs_size);
                let mut result = try!(m.slice_from_mut(lhs_start));
                let &lhs = try!(result.at(idx));

                let mut m = setup::rand::mat::<$ty>(rhs_size);
                let rhs = try!(m.slice_from_mut(rhs_start));

                result.sub_assign(&rhs);

                let &rhs = try!(rhs.at(idx));

                lhs - rhs == *try!(result.at(idx))
            })
        }})+
    }
}

blas!(f32, f64, c64, c128);
