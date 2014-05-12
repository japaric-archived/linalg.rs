use mat;
use num::complex::Cmplx;
use std::num::pow;
use super::super::test::Bencher;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! from_fn {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let size = pow(10.0, $size).sqrt() as uint;
            let size = (size, size);

            b.iter(|| {
                mat::from_fn(size, |i, j| i as $ty - j as $ty)
            })
        }
    }
}

from_fn!(f32_2, 2, f32)
from_fn!(f32_3, 3, f32)
from_fn!(f32_4, 4, f32)
from_fn!(f32_5, 5, f32)
from_fn!(f32_6, 6, f32)

from_fn!(f64_2, 2, f64)
from_fn!(f64_3, 3, f64)
from_fn!(f64_4, 4, f64)
from_fn!(f64_5, 5, f64)
from_fn!(f64_6, 6, f64)

macro_rules! from_fn_cmplx {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let size = pow(10.0, $size).sqrt() as uint;
            let size = (size, size);

            b.iter(|| {
                mat::from_fn(size, |i, j| {
                    Cmplx::new(i as $ty, 0 as $ty) -
                    Cmplx::new(0 as $ty, j as $ty)
                })
            })
        }
    }
}

from_fn_cmplx!(c64_2, 2, f32)
from_fn_cmplx!(c64_3, 3, f32)
from_fn_cmplx!(c64_4, 4, f32)
from_fn_cmplx!(c64_5, 5, f32)
from_fn_cmplx!(c64_6, 6, f32)

from_fn_cmplx!(c128_2, 2, f64)
from_fn_cmplx!(c128_3, 3, f64)
from_fn_cmplx!(c128_4, 4, f64)
from_fn_cmplx!(c128_5, 5, f64)
from_fn_cmplx!(c128_6, 6, f64)
