mod col;

mod trans {
    macro_rules! blas {
        ($($ty:ident),+) => {$(mod $ty {
            use linalg::prelude::*;
            use onezero::Zero;
            use quickcheck::TestResult;

            use setup;

            // Test that `mul(Trans<Mat>)` is correct for `Trans<Mat>`
            #[quickcheck]
            fn mat((m, k, n): (uint, uint, uint), (row, col): (uint, uint)) -> TestResult {
                enforce! {
                    k != 0,
                    row < m,
                    col < n,
                }

                test!({
                    let lhs = setup::rand::mat::<$ty>((k, m)).t();
                    let r = try!(lhs.row(row));

                    let rhs = setup::rand::mat::<$ty>((n, k)).t();
                    let c = try!(rhs.col(col));

                    let result = &lhs * &rhs;
                    let _0: $ty = Zero::zero();
                    let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    product == *try!(result.at((row, col)))
                })
            }

            // Test that `mul(Trans<View>)` is correct for `Trans<Mat>`
            #[quickcheck]
            fn view(
                start: (uint, uint),
                (m, k, n): (uint, uint, uint),
                (row, col): (uint, uint),
            ) -> TestResult {
                enforce! {
                    k != 0,
                    row < m,
                    col < n,
                }

                let size = (start.0 + n, start.1 + k);
                test!({
                    let lhs = setup::rand::mat::<$ty>((k, m)).t();
                    let r = try!(lhs.row(row));

                    let m = setup::rand::mat::<$ty>(size);
                    let rhs = try!(m.slice_from(start)).t();
                    let c = try!(rhs.col(col));

                    let result = &lhs * rhs;
                    let _0: $ty = Zero::zero();
                    let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    product == *try!(result.at((row, col)))
                })
            }

            // Test that `mul(Trans<MutView>)` is correct for `Trans<Mat>`
            #[quickcheck]
            fn view_mut(
                start: (uint, uint),
                (m, k, n): (uint, uint, uint),
                (row, col): (uint, uint),
            ) -> TestResult {
                enforce! {
                    k != 0,
                    row < m,
                    col < n,
                }

                let size = (start.0 + n, start.1 + k);
                test!({
                    let lhs = setup::rand::mat::<$ty>((k, m)).t();
                    let r = try!(lhs.row(row));

                    let mut m = setup::rand::mat::<$ty>(size);
                    let rhs = try!(m.slice_from_mut(start)).t();
                    let c = try!(rhs.col(col));

                    let result = &lhs * &rhs;
                    let _0: $ty = Zero::zero();
                    let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    product == *try!(result.at((row, col)))
                })
            }})+
        }
    }

    blas!(f32, f64, c64, c128);
}

macro_rules! blas {
    ($($ty:ident),+) => {$(mod $ty {
        use linalg::prelude::*;
        use onezero::Zero;
        use quickcheck::TestResult;

        use setup;

        // Test that `mul(Mat)` is correct for `Trans<Mat>`
        #[quickcheck]
        fn mat((m, k, n): (uint, uint, uint), (row, col): (uint, uint)) -> TestResult {
            enforce! {
                k != 0,
                row < m,
                col < n,
            }

            test!({
                let lhs = setup::rand::mat::<$ty>((k, m)).t();
                let r = try!(lhs.row(row));

                let rhs = setup::rand::mat::<$ty>((k, n));
                let c = try!(rhs.col(col));

                let result = &lhs * &rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at((row, col)))
            })
        }

        // Test that `mul(View)` is correct for `Trans<Mat>`
        #[quickcheck]
        fn view(
            start: (uint, uint),
            (m, k, n): (uint, uint, uint),
            (row, col): (uint, uint),
        ) -> TestResult {
            enforce! {
                k != 0,
                row < m,
                col < n,
            }

            let size = (start.0 + k, start.1 + n);
            test!({
                let lhs = setup::rand::mat::<$ty>((k, m)).t();
                let r = try!(lhs.row(row));

                let m = setup::rand::mat::<$ty>(size);
                let rhs = try!(m.slice_from(start));
                let c = try!(rhs.col(col));

                let _0: $ty = Zero::zero();
                let result = &lhs * rhs;
                let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at((row, col)))
            })
        }

        // Test that `mul(MutView)` is correct for `Trans<Mat>`
        #[quickcheck]
        fn view_mut(
            start: (uint, uint),
            (m, k, n): (uint, uint, uint),
            (row, col): (uint, uint),
        ) -> TestResult {
            enforce! {
                k != 0,
                row < m,
                col < n,
            }

            let size = (start.0 + k, start.1 + n);
            test!({
                let lhs = setup::rand::mat::<$ty>((k, m)).t();
                let r = try!(lhs.row(row));

                let mut m = setup::rand::mat::<$ty>(size);
                let rhs = try!(m.slice_from_mut(start));
                let c = try!(rhs.col(col));

                let result = &lhs * &rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at((row, col)))
            })
        }})+
    }
}

blas!(f32, f64, c64, c128);
