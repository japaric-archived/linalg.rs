macro_rules! blas {
    ($($ty:ident),+) => {$(mod $ty {
        mod owned {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `mul(&ColVec)` is correct for `RowVec`
            #[quickcheck]
            fn owned(size: usize) -> TestResult {
                let lhs = setup::rand::row::<$ty>(size);

                let rhs = setup::rand::col::<$ty>(size);

                let result = &lhs * &rhs;
                let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                test!(approx_eq!(result, expected))

            }

            // Test that `mul(Col)` is correct for `RowVec`
            #[quickcheck]
            fn slice((nrows, ncols): (usize, usize), col: usize) -> TestResult {
                enforce! {
                    col < ncols,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((nrows, ncols));
                    let lhs = setup::rand::row::<$ty>(nrows);

                    let rhs = try!(m.col(col));

                    let result = &lhs * rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(&MutCol)` is correct for `RowVec`
            #[quickcheck]
            fn slice_mut((nrows, ncols): (usize, usize), col: usize) -> TestResult {
                enforce! {
                    col < ncols,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let lhs = setup::rand::row::<$ty>(nrows);

                    let rhs = try!(m.col_mut(col));

                    let result = &lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(strided::Col)` is correct for `RowVec`
            #[quickcheck]
            fn strided((nrows, ncols): (usize, usize), col: usize) -> TestResult {
                enforce! {
                    col < ncols,
                }

                test!({
                    let lhs = setup::rand::row::<$ty>(nrows);

                    let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col(col));

                    let result = &lhs * rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(&strided::MutCol)` is correct for `RowVec`
            #[quickcheck]
            fn strided_mut((nrows, ncols): (usize, usize), col: usize) -> TestResult {
                enforce! {
                    col < ncols,
                }

                test!({
                    let lhs = setup::rand::row::<$ty>(nrows);

                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let rhs = try!(m.col_mut(col));

                    let result = &lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }
        }

        mod slice {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `mul(&ColVec)` is correct for `Row`
            #[quickcheck]
            fn owned((nrows, ncols): (usize, usize), row: usize) -> TestResult {
                enforce! {
                    row < nrows,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let lhs = try!(m.row(row));

                    let rhs = setup::rand::col::<$ty>(ncols);

                    let result = lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(Col)` is correct for `Row`
            #[quickcheck]
            fn slice((m, k, n): (usize, usize, usize), (row, col): (usize, usize)) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((k, m)).t();
                    let lhs = try!(m.row(row));

                    let m = setup::rand::mat::<$ty>((k, n));
                    let rhs = try!(m.col(col));

                    let result = lhs * rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(&MutCol)` is correct for `Row`
            #[quickcheck]
            fn slice_mut(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((k, m)).t();
                    let lhs = try!(m.row(row));

                    let mut m = setup::rand::mat::<$ty>((k, n));
                    let rhs = try!(m.col_mut(col));

                    let result = lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(strided::Col)` is correct for `Row`
            #[quickcheck]
            fn strided(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((k, m)).t();
                    let lhs = try!(m.row(row));

                    let m = setup::rand::mat::<$ty>((n, k)).t();
                    let rhs = try!(m.col(col));

                    let result = lhs * rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(&strided::MutCol)` is correct for `Row`
            #[quickcheck]
            fn strided_mut(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((k, m)).t();
                    let lhs = try!(m.row(row));

                    let mut m = setup::rand::mat::<$ty>((n, k)).t();
                    let rhs = try!(m.col_mut(col));

                    let result = lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }
        }

        mod slice_mut {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `mul(&ColVec)` is correct for `MutRow`
            #[quickcheck]
            fn owned((nrows, ncols): (usize, usize), row: usize) -> TestResult {
                enforce! {
                    row < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((ncols, nrows)).t();
                    let lhs = try!(m.row_mut(row));

                    let rhs = setup::rand::col::<$ty>(ncols);

                    let result = &lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(Col)` is correct for `MutRow`
            #[quickcheck]
            fn slice((m, k, n): (usize, usize, usize), (row, col): (usize, usize)) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((k, m)).t();
                    let lhs = try!(m.row_mut(row));

                    let m = setup::rand::mat::<$ty>((k, n));
                    let rhs = try!(m.col(col));

                    let result = &lhs * rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(&MutCol)` is correct for `MutRow`
            #[quickcheck]
            fn slice_mut(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((k, m)).t();
                    let lhs = try!(m.row_mut(row));

                    let mut m = setup::rand::mat::<$ty>((k, n));
                    let rhs = try!(m.col_mut(col));

                    let result = &lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(strided::Col)` is correct for `MutRow`
            #[quickcheck]
            fn strided(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((k, m)).t();
                    let lhs = try!(m.row_mut(row));

                    let m = setup::rand::mat::<$ty>((n, k)).t();
                    let rhs = try!(m.col(col));

                    let result = &lhs * rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(&strided::MutCol)` is correct for `MutRow`
            #[quickcheck]
            fn strided_mut(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((k, m)).t();
                    let lhs = try!(m.row_mut(row));

                    let mut m = setup::rand::mat::<$ty>((n, k)).t();
                    let rhs = try!(m.col_mut(col));

                    let result = &lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }
        }

        mod strided {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `mul(&ColVec)` is correct for `strided::Row`
            #[quickcheck]
            fn owned((nrows, ncols): (usize, usize), row: usize) -> TestResult {
                enforce! {
                    row < nrows,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((nrows, ncols));
                    let lhs = try!(m.row(row));

                    let rhs = setup::rand::col::<$ty>(ncols);

                    let result = lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(Col)` is correct for `strided::Row`
            #[quickcheck]
            fn slice((m, k, n): (usize, usize, usize), (row, col): (usize, usize)) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((m, k));
                    let lhs = try!(m.row(row));

                    let m = setup::rand::mat::<$ty>((k, n));
                    let rhs = try!(m.col(col));

                    let result = lhs * rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(&MutCol)` is correct for `strided::Row`
            #[quickcheck]
            fn slice_mut(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((m, k));
                    let lhs = try!(m.row(row));

                    let mut m = setup::rand::mat::<$ty>((k, n));
                    let rhs = try!(m.col_mut(col));

                    let result = lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(strided::Col)` is correct for `strided::Row`
            #[quickcheck]
            fn strided(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((m, k));
                    let lhs = try!(m.row(row));

                    let m = setup::rand::mat::<$ty>((n, k)).t();
                    let rhs = try!(m.col(col));

                    let result = lhs * rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(&strided::MutCol)` is correct for `strided::Row`
            #[quickcheck]
            fn strided_mut(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let m = setup::rand::mat::<$ty>((m, k));
                    let lhs = try!(m.row(row));

                    let mut m = setup::rand::mat::<$ty>((n, k)).t();
                    let rhs = try!(m.col_mut(col));

                    let result = lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }
        }

        mod strided_mut {
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `mul(&ColVec)` is correct for `strided::MutRow`
            #[quickcheck]
            fn owned((nrows, ncols): (usize, usize), row: usize) -> TestResult {
                enforce! {
                    row < nrows,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((nrows, ncols));
                    let lhs = try!(m.row_mut(row));

                    let rhs = setup::rand::col::<$ty>(ncols);

                    let result = &lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(Col)` is correct for `strided::MutRow`
            #[quickcheck]
            fn slice((m, k, n): (usize, usize, usize), (row, col): (usize, usize)) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((m, k));
                    let lhs = try!(m.row_mut(row));

                    let m = setup::rand::mat::<$ty>((k, n));
                    let rhs = try!(m.col(col));

                    let result = &lhs * rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(&MutCol)` is correct for `strided::MutRow`
            #[quickcheck]
            fn slice_mut(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((m, k));
                    let lhs = try!(m.row_mut(row));

                    let mut m = setup::rand::mat::<$ty>((k, n));
                    let rhs = try!(m.col_mut(col));

                    let result = &lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(strided::Col)` is correct for `strided::MutRow`
            #[quickcheck]
            fn strided(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((m, k));
                    let lhs = try!(m.row_mut(row));

                    let m = setup::rand::mat::<$ty>((n, k)).t();
                    let rhs = try!(m.col(col));

                    let result = &lhs * rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }

            // Test that `mul(&strided::MutCol)` is correct for `strided::MutRow`
            #[quickcheck]
            fn strided_mut(
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    row < m,
                    col < n,
                }

                test!({
                    let mut m = setup::rand::mat::<$ty>((m, k));
                    let lhs = try!(m.row_mut(row));

                    let mut m = setup::rand::mat::<$ty>((n, k)).t();
                    let rhs = try!(m.col_mut(col));

                    let result = &lhs * &rhs;
                    let expected = lhs.iter().zip(rhs.iter()).fold(0., |s, (&x, &y)| s + x * y);

                    approx_eq!(result, expected)
                })
            }
        }})+
    }
}

blas!(f32, f64);
