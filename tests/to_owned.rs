#![allow(unstable)]
#![feature(plugin)]

extern crate linalg;
extern crate quickcheck;
#[plugin]
extern crate quickcheck_macros;

#[macro_use]
mod setup;

mod col {
    macro_rules! blas {
        ($($ty:ident),+) => {$(mod $ty {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `to_owned()` is correct for `ColVec`
            #[quickcheck]
            fn owned(size: usize, idx: usize) -> TestResult {
                enforce! {
                    idx < size,
                }

                test!({
                    let lhs = setup::rand::col::<$ty>(size);

                    let rhs = lhs.to_owned();

                    (try!(lhs.at(idx))) == try!(rhs.at(idx))
                })
            }

            // Test that `to_owned()` is correct for `Col`
            #[quickcheck]
            fn slice((nrows, ncols): (usize, usize), col: usize, idx: usize) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((nrows, ncols));
                    let lhs = try!(m.col(col));

                    let rhs = lhs.to_owned();

                    (try!(lhs.at(idx))) == try!(rhs.at(idx))
                })
            }

            // Test that `to_owned()` is correct for `MutCol`
            #[quickcheck]
            fn slice_mut((nrows, ncols): (usize, usize), col: usize, idx: usize) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let lhs = try!(m.col_mut(col));

                    let rhs = lhs.to_owned();

                    (try!(lhs.at(idx))) == try!(rhs.at(idx))
                })
            }

            // Test that `to_owned()` is correct for `strided::Col`
            #[quickcheck]
            fn strided((nrows, ncols): (usize, usize), col: usize, idx: usize) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let lhs = try!(m.col(col));

                    let rhs = lhs.to_owned();

                    (try!(lhs.at(idx))) == try!(rhs.at(idx))
                })
            }

            // Test that `to_owned()` is correct for `strided::MutCol`
            #[quickcheck]
            fn strided_mut((nrows, ncols): (usize, usize), col: usize, idx: usize) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let lhs = try!(m.col_mut(col));

                    let rhs = lhs.to_owned();

                    (try!(lhs.at(idx))) == try!(rhs.at(idx))
                })
            }})+
        }
    }

    blas!(f32, f64, c64, c128);
}

mod row {
    macro_rules! blas {
        ($($ty:ident),+) => {$(mod $ty {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `to_owned()` is correct for `RowVec`
            #[quickcheck]
            fn owned(size: usize, idx: usize) -> TestResult {
                enforce! {
                    idx < size,
                }

                test!({
                    let lhs = setup::rand::row::<$ty>(size);

                    let rhs = lhs.to_owned();

                    (try!(lhs.at(idx))) == try!(rhs.at(idx))
                })
            }

            // Test that `to_owned()` is correct for `Row`
            #[quickcheck]
            fn slice((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                enforce! {
                    row < nrows,
                    idx < ncols,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let lhs = try!(m.row(row));

                    let rhs = lhs.to_owned();

                    (try!(lhs.at(idx))) == try!(rhs.at(idx))
                })
            }

            // Test that `to_owned()` is correct for `MutRow`
            #[quickcheck]
            fn slice_mut((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                enforce! {
                    row < nrows,
                    idx < ncols,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let lhs = try!(m.row_mut(row));

                    let rhs = lhs.to_owned();

                    (try!(lhs.at(idx))) == try!(rhs.at(idx))
                })
            }

            // Test that `to_owned()` is correct for `strided::Row`
            #[quickcheck]
            fn strided((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                enforce! {
                    row < nrows,
                    idx < ncols,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((nrows, ncols));
                    let lhs = try!(m.row(row));

                    let rhs = lhs.to_owned();

                    (try!(lhs.at(idx))) == try!(rhs.at(idx))
                })
            }

            // Test that `to_owned()` is correct for `strided::MutRow`
            #[quickcheck]
            fn strided_mut((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                enforce! {
                    row < nrows,
                    idx < ncols,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let lhs = try!(m.row_mut(row));

                    let rhs = lhs.to_owned();

                    (try!(lhs.at(idx))) == try!(rhs.at(idx))
                })
            }})+
        }
    }

    blas!(f32, f64, c64, c128);
}

mod trans {
    macro_rules! blas {
        ($($ty:ident),+) => {$(mod $ty {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `to_owned()` is correct for `Trans<Mat>`
            #[quickcheck]
            fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
                enforce! {
                    row < nrows,
                    col < ncols,
                }

                test!({
                    let lhs = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = lhs.to_owned();

                    (try!(lhs.at((row, col)))) == try!(rhs.at((row, col)))
                })
            }

            // Test that `to_owned()` is correct for `Trans<View>`
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
                test!({
                    let m = setup::rand::mat::<$ty>(size);
                    let lhs = try!(m.slice_from(start)).t();
                    let rhs = lhs.to_owned();

                    (try!(lhs.at((row, col)))) == try!(rhs.at((row, col)))
                })
            }

            // Test that `to_owned()` is correct for `Trans<MutView>`
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
                test!({
                    let mut m = setup::rand::mat::<$ty>(size);
                    let lhs = try!(m.slice_from_mut(start)).t();
                    let rhs = lhs.to_owned();

                    (try!(lhs.at((row, col)))) == try!(rhs.at((row, col)))
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

        // Test that `to_owned()` is correct for `Mat`
        #[quickcheck]
        fn mat((nrows, ncols): (usize, usize), (row, col): (usize, usize)) -> TestResult {
            enforce! {
                row < nrows,
                col < ncols,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((nrows, ncols));
                let rhs = lhs.to_owned();

                (try!(lhs.at((row, col)))) == try!(rhs.at((row, col)))
            })
        }

        // Test that `to_owned()` is correct for `View`
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
            test!({
                let m = setup::rand::mat::<$ty>(size);
                let lhs = try!(m.slice_from(start));
                let rhs = lhs.to_owned();

                (try!(lhs.at((row, col)))) == try!(rhs.at((row, col)))
            })
        }

        // Test that `to_owned()` is correct for `MutView`
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
            test!({
                let mut m = setup::rand::mat::<$ty>(size);
                let lhs = try!(m.slice_from_mut(start));
                let rhs = lhs.to_owned();

                (try!(lhs.at((row, col)))) == try!(rhs.at((row, col)))
            })
        }})+
    }
}

blas!(f32, f64, c64, c128);
