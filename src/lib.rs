#![feature(macro_rules, phase)]

extern crate num;
extern crate rand;

#[cfg(test)] #[phase(plugin)] extern crate quickcheck_macros;
#[cfg(test)] extern crate quickcheck;

// XXX There must be a better way to share macros between modules...
macro_rules! assert_shape {
    ($method:ident, $op:tt) => ({
        assert!(self.shape() == rhs.shape(),
                "{}: dimension mismatch: {} {} {}",
                stringify!($method),
                self.shape(),
                stringify!($op),
                rhs.shape());
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
