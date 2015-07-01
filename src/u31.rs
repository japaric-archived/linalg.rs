//! 31-bit unsigned integer

use std::num::{One, Zero};
use std::ops::{Add, AddAssign, Rem};

use cast::From;
use extract::Extract;

/// 31-bit unsigned integer
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct U31(i32);

impl U31 {
    /// Checked integer subtraction
    pub fn checked_sub(self, y: i32) -> Option<U31> {
        let z = self.0 - y;

        if z < 0 {
            None
        } else {
            Some(U31(z))
        }
    }

    /// Casts to `i32`
    pub fn i32(&self) -> i32 {
        self.0
    }

    /// Casts to `isize`
    pub fn isize(&self) -> isize {
        isize::from(self.i32())
    }

    /// Returns the maximum value that U31 can hold
    pub fn max_value() -> U31 {
        U31(i32::max_value())
    }

    /// Casts to `u32`
    pub fn u32(&self) -> u32 {
        unsafe {
            u32::from(self.0).extract()
        }
    }

    /// Casts to `usize`
    pub fn usize(&self) -> usize {
        usize::from(self.u32())
    }
}

impl Add<i32> for U31 {
    type Output = U31;

    fn add(self, y: i32) -> U31 {
        U31(self.0 + y)
    }
}

impl AddAssign<i32> for U31 {
    fn add_assign(&mut self, y: i32) {
        self.0 += y;
    }
}

impl From<i32> for U31 {
    type Output = Option<U31>;

    fn from(z: i32) -> Option<U31> {
        if z < 0 {
            None
        } else {
            Some(U31(z))
        }
    }
}

impl From<u32> for U31 {
    type Output = Option<U31>;

    fn from(z: u32) -> Option<U31> {
        i32::from(z).map(U31)
    }
}

impl From<usize> for U31 {
    type Output = Option<U31>;

    fn from(z: usize) -> Option<U31> {
        i32::from(z).map(U31)
    }
}

impl One for U31 {
    fn one() -> U31 {
        U31(1)
    }
}

impl Rem for U31 {
    type Output = U31;

    fn rem(self, y: U31) -> U31 {
        U31(self.i32() % y.i32())
    }
}

impl Zero for U31 {
    fn zero() -> U31 {
        U31(0)
    }
}
