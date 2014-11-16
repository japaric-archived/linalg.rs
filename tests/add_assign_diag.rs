#![feature(globs, macro_rules, phase)]

extern crate linalg;
extern crate quickcheck;
#[phase(plugin)]
extern crate quickcheck_macros;

mod setup;

macro_rules! blas {
    ($($ty:ident),+) => {$(mod $ty {
        use linalg::prelude::*;
        use quickcheck::TestResult;

        use setup;

        // Test that `add_assign(T)` is correct for `Diag<strided::MutSlice>`
        #[quickcheck]
        fn scalar(size: (uint, uint), diag: int, idx: uint) -> TestResult {
            validate_diag_index!(diag, size, idx)

            test!({
                let mut m = setup::rand::mat::<$ty>(size);
                let mut result = try!(m.diag_mut(diag));
                let &lhs = try!(result.at(idx));

                let rhs: $ty = ::std::rand::random();

                result.add_assign(&rhs);

                lhs + rhs == *try!(result.at(idx))
            })
        }})+
    }
}

blas!(f32, f64, c64, c128)
