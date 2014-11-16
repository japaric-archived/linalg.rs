//! Complex numbers

// TODO Bind to complex functions in libm
// TODO Branch this module into its own crate

use std::fmt;
use std::rand::{Rand, Rng};

use traits::{One, Zero};

/// FFI safe complex number
#[repr(C)]
#[deriving(PartialEq)]
pub struct Complex<T> {
    re: T,
    im: T,
}

#[allow(non_camel_case_types)]
pub type c64 = Complex<f32>;
#[allow(non_camel_case_types)]
pub type c128 = Complex<f64>;

impl<T> Complex<T> {
    /// Returns the complex number `re + j * im`
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex {
            im: im,
            re: re,
        }
    }
}

impl<T> Add<Complex<T>, Complex<T>> for Complex<T> where T: Add<T, T> {
    fn add(&self, rhs: &Complex<T>) -> Complex<T> {
        Complex {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T> Mul<Complex<T>, Complex<T>> for Complex<T> where T: Add<T, T> + Mul<T, T> + Sub<T, T> {
    fn mul(&self, rhs: &Complex<T>) -> Complex<T> {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T> One for Complex<T> where T: Mul<T, T> + One + Sub<T, T> + Zero {
    fn one() -> Complex<T> {
        Complex {
            re: One::one(),
            im: Zero::zero(),
        }
    }
}

impl<T> Rand for Complex<T> where T: Rand {
    fn rand<R>(rng: &mut R) -> Complex<T> where R: Rng {
        Complex::new(rng.gen(), rng.gen())
    }
}

impl<T> fmt::Show for Complex<T> where T: PartialOrd + fmt::Show + Zero {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.im < Zero::zero() {
            write!(f, "{}-{}i", self.re, self.im)
        } else {
            write!(f, "{}+{}i", self.re, self.im)
        }
    }
}

impl<T> Zero for Complex<T> where T: Zero {
    fn zero() -> Complex<T> {
        Complex {
            re: Zero::zero(),
            im: Zero::zero(),
        }
    }
}
