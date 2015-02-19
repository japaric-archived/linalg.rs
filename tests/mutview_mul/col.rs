macro_rules! blas {
    ($ty:ident) => {
        mod $ty {
            use linalg::prelude::*;
            use onezero::Zero;
            use quickcheck::TestResult;

            use setup;

            // Test that `mul(&ColVec)` is correct for `MutView`
            #[quickcheck]
            fn owned(
                start: (usize, usize),
                (m, k): (usize, usize),
                idx: usize,
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

            // Test that `mul(Col)` is correct for `MutView`
            #[quickcheck]
            fn slice(
                start: (usize, usize),
                (m, k, n): (usize, usize, usize),
                col: usize,
                idx: usize,
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

            // Test that `mul(&MutCol)` is correct for `MutView`
            #[quickcheck]
            fn slice_mut(
                start: (usize, usize),
                (m, k, n): (usize, usize, usize),
                col: usize,
                idx: usize,
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

            // Test that `mul(strided::Col)` is correct for `MutView`
            #[quickcheck]
            fn strided(
                start: (usize, usize),
                (m, k, n): (usize, usize, usize),
                col: usize,
                idx: usize,
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

            // Test that `mul(&strided::MutCol)` is correct for `MutView`
            #[quickcheck]
            fn strided_mut(
                start: (usize, usize),
                (m, k, n): (usize, usize, usize),
                col: usize,
                idx: usize,
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
            }
        }
    }
}

blas!(f32);
blas!(f64);
blas!(c64);
blas!(c128);
