macro_rules! blas {
    ($($ty:ident),+) => {$(mod $ty {
        use linalg::prelude::*;
        use onezero::Zero;
        use quickcheck::TestResult;

        use setup;

        // Test that `mul(Col<Box<[T]>>)` is correct for `Mat<T>`
        #[quickcheck]
        fn owned((nrows, ncols): (uint, uint), idx: uint) -> TestResult {
            enforce!{
                idx < nrows,
                ncols != 0,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((nrows, ncols));
                let r = try!(lhs.row(idx));

                let rhs = setup::rand::col::<$ty>(ncols);

                let result = &lhs * &rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                assert_eq!(product , *try!(result.at(idx)))
                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(Col<&[T]>)` is correct for `Mat<T>`
        #[quickcheck]
        fn slice((m, k, n): (uint, uint, uint), col: uint, idx: uint) -> TestResult {
            enforce!{
                col < n,
                idx < m,
                k != 0,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((m, k));
                let r = try!(lhs.row(idx));

                let m = setup::rand::mat::<$ty>((k, n));
                let rhs = try!(m.col(col));

                let result = &lhs * rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                assert_eq!(product , *try!(result.at(idx)))
                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(Col<&mut [T]>)` is correct for `Mat<T>`
        #[quickcheck]
        fn slice_mut((m, k, n): (uint, uint, uint), col: uint, idx: uint) -> TestResult {
            enforce!{
                col < n,
                idx < m,
                k != 0,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((m, k));
                let r = try!(lhs.row(idx));

                let mut m = setup::rand::mat::<$ty>((k, n));
                let rhs = try!(m.col_mut(col));

                let result = &lhs * &rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                assert_eq!(product , *try!(result.at(idx)))
                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(Col<strided::Slice>)` is correct for `Mat<T>`
        #[quickcheck]
        fn strided((m, k, n): (uint, uint, uint), col: uint, idx: uint) -> TestResult {
            enforce!{
                col < n,
                idx < m,
                k != 0,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((m, k));
                let r = try!(lhs.row(idx));

                let m = setup::rand::mat::<$ty>((n, k)).t();
                let rhs = try!(m.col(col));

                let result = &lhs * rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                assert_eq!(product , *try!(result.at(idx)))
                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(Col<strided::MutSlice>)` is correct for `Mat<T>`
        #[quickcheck]
        fn strided_mut((m, k, n): (uint, uint, uint), col: uint, idx: uint) -> TestResult {
            enforce!{
                col < n,
                idx < m,
                k != 0,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((m, k));
                let r = try!(lhs.row(idx));

                let mut m = setup::rand::mat::<$ty>((n, k)).t();
                let rhs = try!(m.col_mut(col));

                let result = &lhs * &rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                assert_eq!(product , *try!(result.at(idx)))
                product == *try!(result.at(idx))
            })
        }})+
    }
}

blas!(f32, f64, c64, c128)
