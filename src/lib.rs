#![feature(macro_rules, phase)]

extern crate num;
extern crate rand;

#[cfg(test)] #[phase(plugin)] extern crate quickcheck_macros;
#[cfg(test)] extern crate quickcheck;

// XXX There must be a better way to share macros between modules...
macro_rules! assert_shape {
    ($lhs:ident, $rhs:ident, $method:ident, $op:tt) => ({
        assert!($lhs.shape() == $rhs.shape(),
                "{}: dimension mismatch: {} {} {}",
                stringify!($method),
                $lhs.shape(),
                stringify!($op),
                $rhs.shape());
    })
}

pub mod array;
pub mod blas;
pub mod common;
pub mod mat;
pub mod vec;

mod traits;

#[cfg(test)] mod bench;
#[cfg(test)] mod test;
