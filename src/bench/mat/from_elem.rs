use mat;
use num::complex::Complex;
use std::rand::random;
use std::num::pow;
use super::super::test::Bencher;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! from_elem {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let size = pow(10f64, $size).sqrt() as uint;
            let size = (size, size);
            let elem = random::<$ty>();

            b.iter(|| {
                mat::from_elem(size, elem)
            })
        }
    }
}

from_elem!(f32_2, 2u, f32)
from_elem!(f32_3, 3u, f32)
from_elem!(f32_4, 4u, f32)
from_elem!(f32_5, 5u, f32)
from_elem!(f32_6, 6u, f32)

from_elem!(f64_2, 2u, f64)
from_elem!(f64_3, 3u, f64)
from_elem!(f64_4, 4u, f64)
from_elem!(f64_5, 5u, f64)
from_elem!(f64_6, 6u, f64)

macro_rules! from_elem_complex {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let size = pow(10f64, $size).sqrt() as uint;
            let size = (size, size);
            let elem = Complex::new(random::<$ty>(), random::<$ty>());

            b.iter(|| {
                mat::from_elem(size, elem)
            })
        }
    }
}

from_elem_complex!(c64_2, 2u, f32)
from_elem_complex!(c64_3, 3u, f32)
from_elem_complex!(c64_4, 4u, f32)
from_elem_complex!(c64_5, 5u, f32)
from_elem_complex!(c64_6, 6u, f32)

from_elem_complex!(c128_2, 2u, f64)
from_elem_complex!(c128_3, 3u, f64)
from_elem_complex!(c128_4, 4u, f64)
from_elem_complex!(c128_5, 5u, f64)
from_elem_complex!(c128_6, 6u, f64)
