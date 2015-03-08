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
            use linalg::prelude::*;
            use quickcheck::TestResult;

            use setup;

            // Test that `sub_assign(&T)` is correct for `MutDiag`
            #[quickcheck]
            fn scalar(size: (usize, usize), diag: isize, idx: usize) -> TestResult {
                validate_diag_index!(diag, size, idx);

                test!({
                    let mut m = setup::rand::mat::<$ty>(size);
                    let mut result = try!(m.diag_mut(diag));
                    let &lhs = try!(result.at(idx));

                    let rhs: $ty = ::rand::random();

                    result.sub_assign(&rhs);

                    approx_eq!(lhs - rhs, *try!(result.at(idx)))
                })
            }
        }
    }
}

blas!(f32);
blas!(f64);
blas!(c64);
// FIXME(#46)
//blas!(c128);
