macro_rules! blas {
    ($($ty:ident),+) => {$(mod $ty {
        use linalg::prelude::*;
        use onezero::Zero;
        use quickcheck::TestResult;

        use setup;

        // Test that `mul(Col<Box<[T]>>)` is correct for `MutView`
        #[quickcheck]
        fn owned(
            start: (uint, uint),
            (m, k): (uint, uint),
            idx: uint,
        ) -> TestResult {
            enforce! {
                idx < m,
                k != 0,
            }

            let size = (start.0 + m, start.1 + k);
            test!({
                let mut m = setup::rand::mat::<$ty>(size);
                let lhs = try!(m.slice_from_mut(start));
                let r = try!(lhs.row(idx));

                let rhs = setup::rand::col::<$ty>(k);

                let result = &lhs * &rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(Col<&[T]>)` is correct for `MutView`
        #[quickcheck]
        fn slice(
            start: (uint, uint),
            (m, k, n): (uint, uint, uint),
            col: uint,
            idx: uint,
        ) -> TestResult {
            enforce! {
                col < n,
                idx < m,
                k != 0,
            }

            let size = (start.0 + m, start.1 + k);
            test!({
                let mut m = setup::rand::mat::<$ty>(size);
                let lhs = try!(m.slice_from_mut(start));
                let r = try!(lhs.row(idx));

                let m = setup::rand::mat::<$ty>((k, n));
                let rhs = try!(m.col(col));

                let result = &lhs * rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(Col<&mut [T]>)` is correct for `MutView`
        #[quickcheck]
        fn slice_mut(
            start: (uint, uint),
            (m, k, n): (uint, uint, uint),
            col: uint,
            idx: uint,
        ) -> TestResult {
            enforce! {
                col < n,
                idx < m,
                k != 0,
            }

            let size = (start.0 + m, start.1 + k);
            test!({
                let mut m = setup::rand::mat::<$ty>(size);
                let lhs = try!(m.slice_from_mut(start));
                let r = try!(lhs.row(idx));

                let mut m = setup::rand::mat::<$ty>((k, n));
                let rhs = try!(m.col_mut(col));

                let result = &lhs * &rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(Col<strided::Slice>)` is correct for `MutView`
        #[quickcheck]
        fn strided(
            start: (uint, uint),
            (m, k, n): (uint, uint, uint),
            col: uint,
            idx: uint,
        ) -> TestResult {
            enforce! {
                col < n,
                idx < m,
                k != 0,
            }

            let size = (start.0 + m, start.1 + k);
            test!({
                let mut m = setup::rand::mat::<$ty>(size);
                let lhs = try!(m.slice_from_mut(start));
                let r = try!(lhs.row(idx));

                let m = setup::rand::mat::<$ty>((n, k)).t();
                let rhs = try!(m.col(col));

                let result = &lhs * rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(rhs.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at(idx))
            })
        }

        // Test that `mul(Col<strided::MutSlice>)` is correct for `MutView`
        #[quickcheck]
        fn strided_mut(
            start: (uint, uint),
            (m, k, n): (uint, uint, uint),
            col: uint,
            idx: uint,
        ) -> TestResult {
            enforce! {
                col < n,
                idx < m,
                k != 0,
            }

            let size = (start.0 + m, start.1 + k);
            test!({
                let mut m = setup::rand::mat::<$ty>(size);
                let lhs = try!(m.slice_from_mut(start));
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
