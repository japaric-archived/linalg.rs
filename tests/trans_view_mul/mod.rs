mod col;

mod trans {
    macro_rules! blas {
        ($($ty:ident),+) => {$(mod $ty {
            use linalg::prelude::*;
            use onezero::Zero;
            use quickcheck::TestResult;

            use setup;

            // Test that `mul(Trans<Mat>)` is correct for `Trans<Viwe>`
            #[quickcheck]
            fn mat(
                start: (uint, uint),
                (m, k, n): (uint, uint, uint),
                (row, col): (uint, uint),
            ) -> TestResult {
                enforce!{
                    k != 0,
                    row < m,
                    col < n,
                }

                let size = (start.0 + k, start.1 + m);
                test!({
                    let m = setup::rand::mat::<$ty>(size);
                    let lhs = try!(m.slice_from(start)).t();
                    let r = try!(lhs.row(row));

                    let rhs = setup::rand::mat::<$ty>((n, k)).t();
                    let c = try!(rhs.col(col));

                    let result = lhs * rhs;
                    let _0: $ty = Zero::zero();
                    let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    product == *try!(result.at((row, col)))
                })
            }

            // Test that `mul(Trans<View>)` is correct for `Trans<View>`
            #[quickcheck]
            fn view(
                (lhs_start, rhs_start): ((uint, uint), (uint, uint)),
                (m, k, n): (uint, uint, uint),
                (row, col): (uint, uint),
            ) -> TestResult {
                enforce!{
                    k != 0,
                    row < m,
                    col < n,
                }

                let lhs_size = (lhs_start.0 + k, lhs_start.1 + m);
                let rhs_size = (rhs_start.0 + n, rhs_start.1 + k);
                test!({
                    let m = setup::rand::mat::<$ty>(lhs_size);
                    let lhs = try!(m.slice_from(lhs_start)).t();
                    let r = try!(lhs.row(row));

                    let m = setup::rand::mat::<$ty>(rhs_size);
                    let rhs = try!(m.slice_from(rhs_start)).t();
                    let c = try!(rhs.col(col));

                    let result = lhs * rhs;
                    let _0: $ty = Zero::zero();
                    let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    product == *try!(result.at((row, col)))
                })
            }

            // Test that `mul(Trans<MutView>)` is correct for `Trans<View>`
            #[quickcheck]
            fn view_mut(
                (lhs_start, rhs_start): ((uint, uint), (uint, uint)),
                (m, k, n): (uint, uint, uint),
                (row, col): (uint, uint),
            ) -> TestResult {
                enforce!{
                    k != 0,
                    row < m,
                    col < n,
                }

                let lhs_size = (lhs_start.0 + k, lhs_start.1 + m);
                let rhs_size = (rhs_start.0 + n, rhs_start.1 + k);
                test!({
                    let m = setup::rand::mat::<$ty>(lhs_size);
                    let lhs = try!(m.slice_from(lhs_start)).t();
                    let r = try!(lhs.row(row));

                    let mut m = setup::rand::mat::<$ty>(rhs_size);
                    let rhs = try!(m.slice_from_mut(rhs_start)).t();
                    let c = try!(rhs.col(col));

                    let result = lhs * rhs;
                    let _0: $ty = Zero::zero();
                    let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    product == *try!(result.at((row, col)))
                })
            }})+
        }
    }

    blas!(f32, f64, c64, c128)
}

macro_rules! blas {
    ($($ty:ident),+) => {$(mod $ty {
        use linalg::prelude::*;
        use onezero::Zero;
        use quickcheck::TestResult;

        use setup;

        // Test that `mul(Mat)` is correct for `Trans<View>`
        #[quickcheck]
        fn mat(
            start: (uint, uint),
            (m, k, n): (uint, uint, uint),
            (row, col): (uint, uint),
        ) -> TestResult {
            enforce!{
                k != 0,
                row < m,
                col < n,
            }

            let size = (start.0 + k, start.1 + m);
            test!({
                let m = setup::rand::mat::<$ty>(size);
                let lhs = try!(m.slice_from(start)).t();
                let r = try!(lhs.row(row));

                let rhs = setup::rand::mat::<$ty>((k, n));
                let c = try!(rhs.col(col));

                let result = lhs * rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at((row, col)))
            })
        }

        // Test that `mul(View)` is correct for `Trans<View>`
        #[quickcheck]
        fn view(
            (lhs_start, rhs_start): ((uint, uint), (uint, uint)),
            (m, k, n): (uint, uint, uint),
            (row, col): (uint, uint),
        ) -> TestResult {
            enforce!{
                k != 0,
                row < m,
                col < n,
            }

            let lhs_size = (lhs_start.0 + k, lhs_start.1 + m);
            let rhs_size = (rhs_start.0 + k, rhs_start.1 + n);
            test!({
                let m = setup::rand::mat::<$ty>(lhs_size);
                let lhs = try!(m.slice_from(lhs_start)).t();
                let r = try!(lhs.row(row));

                let m = setup::rand::mat::<$ty>(rhs_size);
                let rhs = try!(m.slice_from(rhs_start));
                let c = try!(rhs.col(col));

                let _0: $ty = Zero::zero();
                let result = lhs * rhs;
                let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at((row, col)))
            })
        }

        // Test that `mul(MutView)` is correct for `Trans<View>`
        #[quickcheck]
        fn view_mut(
            (lhs_start, rhs_start): ((uint, uint), (uint, uint)),
            (m, k, n): (uint, uint, uint),
            (row, col): (uint, uint),
        ) -> TestResult {
            enforce!{
                k != 0,
                row < m,
                col < n,
            }

            let lhs_size = (lhs_start.0 + k, lhs_start.1 + m);
            let rhs_size = (rhs_start.0 + k, rhs_start.1 + n);
            test!({
                let m = setup::rand::mat::<$ty>(lhs_size);
                let lhs = try!(m.slice_from(lhs_start)).t();
                let r = try!(lhs.row(row));

                let mut m = setup::rand::mat::<$ty>(rhs_size);
                let rhs = try!(m.slice_from_mut(rhs_start));
                let c = try!(rhs.col(col));

                let result = lhs * rhs;
                let _0: $ty = Zero::zero();
                let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                product == *try!(result.at((row, col)))
            })
        }})+
    }
}

blas!(f32, f64, c64, c128)
