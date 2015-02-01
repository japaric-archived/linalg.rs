#![feature(plugin)]
#![feature(rand)]

extern crate linalg;
extern crate quickcheck;
#[plugin]
extern crate quickcheck_macros;

#[macro_use]
mod setup;

macro_rules! blas {
    ($($ty:ident),+) => {$(mod $ty {
        use linalg::prelude::*;
        use quickcheck::TestResult;

        use setup;

        // Test that `add_assign(T)` is correct for `MutDiag`
        #[quickcheck]
        fn scalar(size: (usize, usize), diag: isize, idx: usize) -> TestResult {
            validate_diag_index!(diag, size, idx);

            test!({
                let mut m = setup::rand::mat::<$ty>(size);
                let mut result = try!(m.diag_mut(diag));
                let &lhs = try!(result.at(idx));

                let rhs: $ty = ::std::rand::random();

                result.add_assign(rhs);

                lhs + rhs == *try!(result.at(idx))
            })
        }})+
    }
}

blas!(f32, f64, c64, c128);
