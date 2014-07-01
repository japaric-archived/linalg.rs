use num::Complex;
use std::num;

use mat;
use super::super::test::Bencher;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! from_fn {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let size = num::pow(10f64, $size).sqrt() as uint;
            let size = (size, size);

            b.iter(|| {
                mat::from_fn(size, |i, j| i as $ty - j as $ty)
            })
        }
    }
}

from_fn!(f32_2, 2u, f32)
from_fn!(f32_3, 3u, f32)
from_fn!(f32_4, 4u, f32)
from_fn!(f32_5, 5u, f32)
from_fn!(f32_6, 6u, f32)

from_fn!(f64_2, 2u, f64)
from_fn!(f64_3, 3u, f64)
from_fn!(f64_4, 4u, f64)
from_fn!(f64_5, 5u, f64)
from_fn!(f64_6, 6u, f64)

macro_rules! from_fn_complex {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let size = num::pow(10f64, $size).sqrt() as uint;
            let size = (size, size);

            b.iter(|| {
                mat::from_fn(size, |i, j| {
                    Complex::new(i as $ty, num::zero()) -
                    Complex::new(num::zero(), j as $ty)
                })
            })
        }
    }
}

from_fn_complex!(c64_2, 2u, f32)
from_fn_complex!(c64_3, 3u, f32)
from_fn_complex!(c64_4, 4u, f32)
from_fn_complex!(c64_5, 5u, f32)
from_fn_complex!(c64_6, 6u, f32)

from_fn_complex!(c128_2, 2u, f64)
from_fn_complex!(c128_3, 3u, f64)
from_fn_complex!(c128_4, 4u, f64)
from_fn_complex!(c128_5, 5u, f64)
from_fn_complex!(c128_6, 6u, f64)
