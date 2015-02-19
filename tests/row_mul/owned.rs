mod trans {
    macro_rules! blas {
        ($ty:ident) => {
            mod $ty {
                use linalg::prelude::*;
                use onezero::Zero;
                use quickcheck::TestResult;

                use setup;

                // Test that `mul(&Trans<Mat>)` is correct for `RowVec`
                #[quickcheck]
                fn mat((k, n): (usize, usize), idx: usize) -> TestResult {
                    enforce! {
                        k != 0,
                        idx < n,
                    }

                    test!({
                        let lhs = setup::rand::row::<$ty>(k);

                        let rhs = setup::rand::mat::<$ty>((n, k)).t();
                        let c = try!(rhs.col(idx));

                        let result = &lhs * &rhs;

                        let _0: $ty = Zero::zero();
                        let product = lhs.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                        product == *try!(result.at(idx))
                    })
                }

                // Test that `mul(Trans<View>)` is correct for `RowVec`
                #[quickcheck]
                fn view(
                    start: (usize, usize),
                    (k, n): (usize, usize),
                    idx: usize,
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        idx < n,
                    }

                    let size = (start.0 + n, start.1 + k);
                    test!({
                        let lhs = setup::rand::row::<$ty>(k);

                        let m = setup::rand::mat::<$ty>(size);
                        let rhs = try!(m.slice_from(start)).t();
                        let c = try!(rhs.col(idx));

                        let _0: $ty = Zero::zero();
                        let result = &lhs * rhs;
                        let product = lhs.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                        product == *try!(result.at(idx))
                    })
                }

                // Test that `mul(&Trans<MutView>)` is correct for `RowVec`
                #[quickcheck]
                fn view_mut(
                    start: (usize, usize),
                    (k, n): (usize, usize),
                    idx: usize,
                ) -> TestResult {
                    enforce! {
                        k != 0,
                        idx < n,
                    }

                    let size = (start.0 + n, start.1 + k);
                    test!({
                        let lhs = setup::rand::row::<$ty>(k);

                        let mut m = setup::rand::mat::<$ty>(size);
                        let rhs = try!(m.slice_from_mut(start)).t();
                        let c = try!(rhs.col(idx));

                        let _0: $ty = Zero::zero();
                        let result = &lhs * &rhs;
                        let product = lhs.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                        product == *try!(result.at(idx))
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

            // Test that `mul(&Mat)` is correct for `RowVec`
            #[quickcheck]
            fn mat((k, n): (usize, usize), idx: usize) -> TestResult {
                enforce! {
                    k != 0,
                    idx < n,
                }

                test!({
                    let lhs = setup::rand::row::<$ty>(k);

                    let rhs = setup::rand::mat::<$ty>((k, n));
                    let c = try!(rhs.col(idx));

                    let result = &lhs * &rhs;

                    let _0: $ty = Zero::zero();
                    let product = lhs.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    product == *try!(result.at(idx))
                })
            }

            // Test that `mul(View)` is correct for `RowVec`
            #[quickcheck]
            fn view(
                start: (usize, usize),
                (k, n): (usize, usize),
                idx: usize,
            ) -> TestResult {
                enforce! {
                    k != 0,
                    idx < n,
                }

                let size = (start.0 + k, start.1 + n);
                test!({
                    let lhs = setup::rand::row::<$ty>(k);

                    let m = setup::rand::mat::<$ty>(size);
                    let rhs = try!(m.slice_from(start));
                    let c = try!(rhs.col(idx));

                    let _0: $ty = Zero::zero();
                    let result = &lhs * rhs;
                    let product = lhs.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    product == *try!(result.at(idx))
                })
            }

            // Test that `mul(&MutView)` is correct for `RowVec`
            #[quickcheck]
            fn view_mut(
                start: (usize, usize),
                (k, n): (usize, usize),
                idx: usize,
            ) -> TestResult {
                enforce! {
                    k != 0,
                    idx < n,
                }

                let size = (start.0 + k, start.1 + n);
                test!({
                    let lhs = setup::rand::row::<$ty>(k);

                    let mut m = setup::rand::mat::<$ty>(size);
                    let rhs = try!(m.slice_from_mut(start));
                    let c = try!(rhs.col(idx));

                    let _0: $ty = Zero::zero();
                    let result = &lhs * &rhs;
                    let product = lhs.iter().zip(c.iter()).fold(_0, |s, (&x, &y)| x * y + s);

                    product == *try!(result.at(idx))
                })
            }
        }
    }
}

blas!(f32);
blas!(f64);
blas!(c64);
blas!(c128);
