#![crate_id="linalg#0.11-pre"]
#![crate_type="lib"]
#![feature(macro_rules)]

extern crate num;
extern crate rand;

#[cfg(test)] extern crate quickcheck;
// FIXME No cargo support for quickcheck_macros (yet)
//#[cfg(test)] extern crate quickcheck_macros;

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
