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

            // Test that `sub_assign(Col<Box<[T]>>)` is correct for `Col<Box<[T]>>`
            #[quickcheck]
            fn owned(size: uint, idx: uint) -> TestResult {
                enforce!{
                    idx < size,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(size);
                    let &lhs = try!(result.at(idx));

                    let rhs = setup::rand::col::<$ty>(size);

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(T)` is correct for `Col<Box<[T]>>`
            #[quickcheck]
            fn scalar(size: uint, idx: uint) -> TestResult {
                enforce!{
                    idx < size,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(size);
                    let &lhs = try!(result.at(idx));

                    let rhs: $ty = ::std::rand::random();

                    result.sub_assign(&rhs);

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<&[T]>)` is correct for `Col<Box<[T]>>`
            #[quickcheck]
            fn slice((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(nrows);
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<&mut [T]>)` is correct for `Col<Box<[T]>>`
            #[quickcheck]
            fn slice_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(nrows);
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col_mut(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<strided::Slice>)` is correct for `Col<Box<[T]>>`
            #[quickcheck]
            fn strided((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(nrows);
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<strided::MutSlice>)` is correct for `Col<Box<[T]>>`
            #[quickcheck]
            fn strided_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut result = setup::rand::col::<$ty>(nrows);
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col_mut(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }
        }

        mod slice_mut {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `sub_assign(Col<Box<[T]>>)` is correct for `Col<&mut [T]>`
            #[quickcheck]
            fn owned((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let rhs = setup::rand::col::<$ty>(nrows);

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(T)` is correct for `Col<&mut [T]>`
            #[quickcheck]
            fn scalar((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let rhs: $ty = ::std::rand::random();

                    result.sub_assign(&rhs);

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<&[T]>)` is correct for `Col<&mut [T]>`
            #[quickcheck]
            fn slice((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<&mut [T]>)` is correct for `Col<&mut [T]>`
            #[quickcheck]
            fn slice_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col_mut(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<strided::Slice>)` is correct for `Col<&mut [T]>`
            #[quickcheck]
            fn strided((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<strided::MutSlice>)` is correct for `Col<&mut [T]>`
            #[quickcheck]
            fn strided_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col_mut(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }
        }

        mod strided_mut {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `sub_assign(T)` is correct for `Col<strided::MutSlice>`
            #[quickcheck]
            fn owned((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let rhs = setup::rand::col(nrows);

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(T)` is correct for `Col<strided::MutSlice>`
            #[quickcheck]
            fn scalar((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let rhs: $ty = ::std::rand::random();

                    result.sub_assign(&rhs);

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<&[T]>)` is correct for `Col<strided::MutSlice>`
            #[quickcheck]
            fn slice((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<&mut [T]>)` is correct for `Col<strided::MutSlice>`
            #[quickcheck]
            fn slice_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let rhs = try!(m.col_mut(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<strided::Slice>)` is correct for `Col<strided::MutSlice>`
            #[quickcheck]
            fn strided((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }

            // Test that `sub_assign(Col<strided::Slice>)` is correct for `Col<strided::MutSlice>`
            #[quickcheck]
            fn strided_mut((nrows, ncols): (uint, uint), col: uint, idx: uint) -> TestResult {
                enforce!{
                    col < ncols,
                    idx < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let mut result = try!(m.col_mut(col));
                    let &lhs = try!(result.at(idx));

                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col_mut(col));

                    result.sub_assign(&rhs);

                    let &rhs = try!(rhs.at(idx));

                    lhs - rhs == *try!(result.at(idx))
                })
            }
        }})+
    }
}

blas!(f32, f64, c64, c128)
