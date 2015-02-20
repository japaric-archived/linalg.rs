mod col;

mod trans {
    macro_rules! blas {
        ($ty:ident) => {
            mod $ty {
                use linalg::prelude::*;
                use onezero::Zero;
                use quickcheck::TestResult;

                use setup;

                // Test that `mul(&Trans<Mat>)` is correct for `View`
                #[quickcheck]
                fn mat(
                    start: (usize, usize),
                    (m, k, n): (usize, usize, usize),
                    (row, col): (usize, usize),
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let size = (start.0 + m, start.1 + k);
                    test!({
                        let m = setup::rand::mat::<$ty>(size);
                        let lhs = try!(m.slice(start..));
                        let r = try!(lhs.row(row));

                        let rhs = setup::rand::mat::<$ty>((n, k)).t();
                        let c = try!(rhs.col(col));

                        let result = lhs * &rhs;
                        let _0: $ty = Zero::zero();
                        let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                        approx_eq!(product, *try!(result.at((row, col))))
                    })
                }

                // Test that `mul(Trans<View>)` is correct for `View`
                #[quickcheck]
                fn view(
                    (lhs_start, rhs_start): ((usize, usize), (usize, usize)),
                    (m, k, n): (usize, usize, usize),
                    (row, col): (usize, usize),
                  ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let lhs_size = (lhs_start.0 + m, lhs_start.1 + k);
                    let rhs_size = (rhs_start.0 + n, rhs_start.1 + k);
                    test!({
                        let m = setup::rand::mat::<$ty>(lhs_size);
                        let lhs = try!(m.slice(lhs_start..));
                        let r = try!(lhs.row(row));

                        let m = setup::rand::mat::<$ty>(rhs_size);
                        let rhs = try!(m.slice(rhs_start..)).t();
                        let c = try!(rhs.col(col));

                        let result = lhs * rhs;
                        let _0: $ty = Zero::zero();
                        let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                        approx_eq!(product, *try!(result.at((row, col))))
                    })
                }

                // Test that `mul(&Trans<MutView>)` is correct for `View`
                #[quickcheck]
                fn view_mut(
                    (lhs_start, rhs_start): ((usize, usize), (usize, usize)),
                    (m, k, n): (usize, usize, usize),
                    (row, col): (usize, usize),
                  ) -> TestResult {
                    enforce! {
                        k != 0,
                        row < m,
                        col < n,
                    }

                    let lhs_size = (lhs_start.0 + m, lhs_start.1 + k);
                    let rhs_size = (rhs_start.0 + n, rhs_start.1 + k);
                    test!({
                        let m = setup::rand::mat::<$ty>(lhs_size);
                        let lhs = try!(m.slice(lhs_start..));
                        let r = try!(lhs.row(row));

                        let mut m = setup::rand::mat::<$ty>(rhs_size);
                        let rhs = try!(m.slice_mut(rhs_start..)).t();
                        let c = try!(rhs.col(col));

                        let result = lhs * &rhs;
                        let _0: $ty = Zero::zero();
                        let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                        approx_eq!(product, *try!(result.at((row, col))))
                    })
                }
            }
        }
    }

    blas!(f32);
    blas!(f64);
    blas!(c64);
    blas!(c128);
}

macro_rules! blas {
    ($ty:ident) => {
        mod $ty {
            use linalg::prelude::*;
            use onezero::Zero;
            use quickcheck::TestResult;

            use setup;

            // Test that `mul(&Mat)` is correct for `View`
            #[quickcheck]
            fn mat(
                start: (usize, usize),
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    k != 0,
                    row < m,
                    col < n,
                }

                let size = (start.0 + m, start.1 + k);
                test!({
                    let m = setup::rand::mat::<$ty>(size);
                    let lhs = try!(m.slice(start..));
                    let r = try!(lhs.row(row));

                    let rhs = setup::rand::mat::<$ty>((k, n));
                    let c = try!(rhs.col(col));

                    let result = lhs * &rhs;
                    let _0: $ty = Zero::zero();
                    let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    approx_eq!(product, *try!(result.at((row, col))))
                })
            }

            // Test that `mul(View)` is correct for `View`
            #[quickcheck]
            fn view(
                (lhs_start, rhs_start): ((usize, usize), (usize, usize)),
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    k != 0,
                    row < m,
                    col < n,
                }

                let lhs_size = (lhs_start.0 + m, lhs_start.1 + k);
                let rhs_size = (rhs_start.0 + k, rhs_start.1 + n);
                test!({
                    let m = setup::rand::mat::<$ty>(lhs_size);
                    let lhs = try!(m.slice(lhs_start..));
                    let r = try!(lhs.row(row));

                    let m = setup::rand::mat::<$ty>(rhs_size);
                    let rhs = try!(m.slice(rhs_start..));
                    let c = try!(rhs.col(col));

                    let result = lhs * rhs;
                    let _0: $ty = Zero::zero();
                    let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    approx_eq!(product, *try!(result.at((row, col))))
                })
            }

            // Test that `mul(&MutView)` is correct for `View`
            #[quickcheck]
            fn view_mut(
                (lhs_start, rhs_start): ((usize, usize), (usize, usize)),
                (m, k, n): (usize, usize, usize),
                (row, col): (usize, usize),
            ) -> TestResult {
                enforce! {
                    k != 0,
                    row < m,
                    col < n,
                }

                let lhs_size = (lhs_start.0 + m, lhs_start.1 + k);
                let rhs_size = (rhs_start.0 + k, rhs_start.1 + n);
                test!({
                    let m = setup::rand::mat::<$ty>(lhs_size);
                    let lhs = try!(m.slice(lhs_start..));
                    let r = try!(lhs.row(row));

                    let mut m = setup::rand::mat::<$ty>(rhs_size);
                    let rhs = try!(m.slice_mut(rhs_start..));
                    let c = try!(rhs.col(col));

                    let result = lhs * &rhs;
                    let _0: $ty = Zero::zero();
                    let product = r.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    approx_eq!(product, *try!(result.at((row, col))))
                })
            }
        }
    }
}

blas!(f32);
blas!(f64);
blas!(c64);
blas!(c128);
