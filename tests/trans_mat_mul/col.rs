macro_rules! blas {
    ($($ty:ident),+) => {$(mod $ty {
        use linalg::prelude::*;
        use onezero::Zero;
        use quickcheck::TestResult;

        use setup;

        // Test that `mul(&ColVec)` is correct for `Trans<Mat>`
        #[quickcheck]
        fn owned((nrows, ncols): (uint, uint), idx: uint) -> TestResult {
            enforce! {
                idx < nrows,
                ncols != 0,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((ncols, nrows)).t();
                let r = try!(lhs.row(idx));

                let rhs = setup::rand::col::<$ty>(ncols);

                let result = &lhs * &rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(Col)` is correct for `Trans<Mat>`
        #[quickcheck]
        fn slice((m, k, n): (uint, uint, uint), col: uint, idx: uint) -> TestResult {
            enforce! {
                col < n,
                idx < m,
                k != 0,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((k, m)).t();
                let r = try!(lhs.row(idx));

                let m = setup::rand::mat::<$ty>((k, n));
                let rhs = try!(m.col(col));

                let result = &lhs * rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(&MutCol)` is correct for `Trans<Mat>`
        #[quickcheck]
        fn slice_mut((m, k, n): (uint, uint, uint), col: uint, idx: uint) -> TestResult {
            enforce! {
                col < n,
                idx < m,
                k != 0,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((k, m)).t();
                let r = try!(lhs.row(idx));

                let mut m = setup::rand::mat::<$ty>((k, n));
                let rhs = try!(m.col_mut(col));

                let result = &lhs * &rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(strided::Col)` is correct for `Trans<Mat>`
        #[quickcheck]
        fn strided((m, k, n): (uint, uint, uint), col: uint, idx: uint) -> TestResult {
            enforce! {
                col < n,
                idx < m,
                k != 0,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((k, m)).t();
                let r = try!(lhs.row(idx));

                let m = setup::rand::mat::<$ty>((n, k)).t();
                let rhs = try!(m.col(col));

                let result = &lhs * rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(&strided::MutCol)` is correct for `Trans<Mat>`
        #[quickcheck]
        fn strided_mut((m, k, n): (uint, uint, uint), col: uint, idx: uint) -> TestResult {
            enforce! {
                col < n,
                idx < m,
                k != 0,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((k, m)).t();
                let r = try!(lhs.row(idx));

                let mut m = setup::rand::mat::<$ty>((n, k)).t();
                let rhs = try!(m.col_mut(col));

                let result = &lhs * &rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at(idx))
            })
        }})+
    }
}

blas!(f32, f64, c64, c128);
