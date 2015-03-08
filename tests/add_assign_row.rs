#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate approx;
extern crate linalg;
extern crate quickcheck;
extern crate rand;

#[macro_use]
mod setup;

macro_rules! blas {
    ($ty:ident) => {
        mod $ty {
            mod owned {
                use linalg::prelude::*;
                use quickcheck::TestResult;

                use setup;

                // Test that `add_assign(&RowVec)` is correct for `RowVec`
                #[quickcheck]
                fn owned(size: usize, idx: usize) -> TestResult {
                    enforce! {
                        idx < size,
                    }

                    test!({
                        let mut result = setup::rand::row::<$ty>(size);
                        let &lhs = try!(result.at(idx));

                        let rhs = setup::rand::row::<$ty>(size);

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&T)` is correct for `RowVec`
                #[quickcheck]
                fn scalar(size: usize, idx: usize) -> TestResult {
                    enforce! {
                        idx < size,
                    }

                    test!({
                        let mut result = setup::rand::row::<$ty>(size);
                        let &lhs = try!(result.at(idx));

                        let rhs: $ty = ::rand::random();

                        result.add_assign(&rhs);

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&Row)` is correct for `RowVec`
                #[quickcheck]
                fn slice((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut result = setup::rand::row::<$ty>(ncols);
                        let &lhs = try!(result.at(idx));

                        let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let rhs = try!(m.row(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&MutRow)` is correct for `RowVec`
                #[quickcheck]
                fn slice_mut(
                    (nrows, ncols): (usize, usize),
                    row: usize,
                    idx: usize,
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut result = setup::rand::row::<$ty>(ncols);
                        let &lhs = try!(result.at(idx));

                        let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let rhs = try!(m.row_mut(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&strided::Row)` is correct for `RowVec`
                #[quickcheck]
                fn strided((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut result = setup::rand::row::<$ty>(ncols);
                        let &lhs = try!(result.at(idx));

                        let m = setup::rand::mat::<$ty>((nrows, ncols));
                        let rhs = try!(m.row(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&strided::MutRow)` is correct for `RowVec`
                #[quickcheck]
                fn strided_mut(
                    (nrows, ncols): (usize, usize),
                    row: usize,
                    idx: usize,
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut result = setup::rand::row::<$ty>(ncols);
                        let &lhs = try!(result.at(idx));

                        let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                        let rhs = try!(m.row_mut(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }
            }

            mod slice_mut {
                use linalg::prelude::*;
                use quickcheck::TestResult;

                use setup;

                // Test that `add_assign(&RowVec)` is correct for `MutRow`
                #[quickcheck]
                fn owned((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let rhs = setup::rand::row::<$ty>(ncols);

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&T)` is correct for `MutRow`
                #[quickcheck]
                fn scalar((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let rhs: $ty = ::rand::random();

                        result.add_assign(&rhs);

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&Row)` is correct for `MutRow`
                #[quickcheck]
                fn slice((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let rhs = try!(m.row(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&MutRow)` is correct for `MutRow`
                #[quickcheck]
                fn slice_mut(
                    (nrows, ncols): (usize, usize),
                    row: usize,
                    idx: usize,
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let rhs = try!(m.row_mut(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&strided::Row)` is correct for `MutRow`
                #[quickcheck]
                fn strided((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let m = setup::rand::mat::<$ty>((nrows, ncols));
                        let rhs = try!(m.row(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&strided::MutRow)` is correct for `MutRow`
                #[quickcheck]
                fn strided_mut(
                    (nrows, ncols): (usize, usize),
                    row: usize,
                    idx: usize,
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                        let rhs = try!(m.row_mut(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }
            }

            mod strided_mut {
                use linalg::prelude::*;
                use quickcheck::TestResult;

                use setup;

                // Test that `add_assign(&RowVec)` is correct for `strided::MutRow`
                #[quickcheck]
                fn owned((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let rhs = setup::rand::row(ncols);

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&T)` is correct for `strided::MutRow`
                #[quickcheck]
                fn scalar((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let rhs: $ty = ::rand::random();

                        result.add_assign(&rhs);

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&Row)` is correct for `strided::MutRow`
                #[quickcheck]
                fn slice((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let rhs = try!(m.row(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&MutRow)` is correct for `strided::MutRow`
                #[quickcheck]
                fn slice_mut(
                    (nrows, ncols): (usize, usize),
                    row: usize,
                    idx: usize,
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                        let rhs = try!(m.row_mut(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&strided::Row)` is correct for `strided::MutRow`
                #[quickcheck]
                fn strided((nrows, ncols): (usize, usize), row: usize, idx: usize) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let m = setup::rand::mat::<$ty>((nrows, ncols));
                        let rhs = try!(m.row(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }

                // Test that `add_assign(&strided::MutRow)` is correct for `strided::MutRow`
                #[quickcheck]
                fn strided_mut(
                    (nrows, ncols): (usize, usize),
                    row: usize,
                    idx: usize,
                ) -> TestResult {
                    enforce! {
                        row < nrows,
                        idx < ncols,
                    }

                    test!({
                        let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                        let mut result = try!(m.row_mut(row));
                        let &lhs = try!(result.at(idx));

                        let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                        let rhs = try!(m.row_mut(row));

                        result.add_assign(&rhs);

                        let &rhs = try!(rhs.at(idx));

                        approx_eq!(lhs + rhs, *try!(result.at(idx)))
                    })
                }
            }
        }
    }
}

blas!(f32);
blas!(f64);
blas!(c64);
// FIXME(#46)
//blas!(c128);
