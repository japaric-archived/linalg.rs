#![feature(globs, macro_rules, phase)]

extern crate linalg;
extern crate quickcheck;
#[phase(plugin)]
extern crate quickcheck_macros;

mod setup;

macro_rules! blas {
    ($($ty:ident),+) => {$(mod $ty {
        mod owned {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `add_assign(&ColVec)` is correct for `ColVec`
            #[quickcheck]
            fn owned(size: uint, idx: uint) -> TestResult {
                enforce! {
                    idx < size,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(size);
                    let &lhs = try!(result.at(idx));

                    let rhs = setup::rand::col::<$ty>(size);

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(T)` is correct for `ColVec`
            #[quickcheck]
            fn scalar(size: uint, idx: uint) -> TestResult {
                enforce! {
                    idx < size,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(size);
                    let &lhs = try!(result.at(idx));

                    let rhs: $ty = ::std::rand::random();

                    result.add_assign(rhs);

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(Col)` is correct for `ColVec`
            #[quickcheck]
            fn slice((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(nrows);
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col(col));

                    result.add_assign(rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(&MutCol)` is correct for `ColVec`
            #[quickcheck]
            fn slice_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(nrows);
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col_mut(col));

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(strided::Col)` is correct for `ColVec`
            #[quickcheck]
            fn strided((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(nrows);
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col(col));

                    result.add_assign(rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(&strided::MutCol)` is correct for `ColVec`
            #[quickcheck]
            fn strided_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(nrows);
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col_mut(col));

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }
        }

        mod slice_mut {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `add_assign(&ColVec)` is correct for `MutCol`
            #[quickcheck]
            fn owned((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let rhs = setup::rand::col::<$ty>(nrows);

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(T)` is correct for `MutCol`
            #[quickcheck]
            fn scalar((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let rhs: $ty = ::std::rand::random();

                    result.add_assign(rhs);

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(Col)` is correct for `MutCol`
            #[quickcheck]
            fn slice((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col(col));

                    result.add_assign(rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(&MutCol)` is correct for `MutCol`
            #[quickcheck]
            fn slice_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col_mut(col));

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(strided::Col)` is correct for `MutCol`
            #[quickcheck]
            fn strided((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col(col));

                    result.add_assign(rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(&strided::MutCol)` is correct for `MutCol`
            #[quickcheck]
            fn strided_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col_mut(col));

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }
        }

        mod strided_mut {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `add_assign(&ColVec)` is correct for `strided::MutCol`
            #[quickcheck]
            fn owned((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let rhs = setup::rand::col(nrows);

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(T)` is correct for `strided::MutCol`
            #[quickcheck]
            fn scalar((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let rhs: $ty = ::std::rand::random();

                    result.add_assign(rhs);

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(Col)` is correct for `strided::MutCol`
            #[quickcheck]
            fn slice((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col(col));

                    result.add_assign(rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(&MutCol)` is correct for `strided::MutCol`
            #[quickcheck]
            fn slice_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col_mut(col));

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(strided::Col)` is correct for `strided::MutCol`
            #[quickcheck]
            fn strided((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col(col));

                    result.add_assign(rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }

            // Test that `add_assign(&strided::MutCol)` is correct for `strided::MutCol`
            #[quickcheck]
            fn strided_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce! {
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col_mut(col));

                    result.add_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs + rhs == *try!(result.at(idx))
                })
            }
        }})+
    }
}

blas!(f32, f64, c64, c128);
