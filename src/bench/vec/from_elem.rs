use num::Complex;
use std::{num,rand};

use super::super::test::Bencher;
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! from_elem {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let size = num::pow(10u, $size);
            let elem = rand::random::<$ty>();

            b.iter(|| {
                vec::from_elem(size, elem)
            })
        }
    }
}

from_elem!(f32_2, 2, f32)
from_elem!(f32_3, 3, f32)
from_elem!(f32_4, 4, f32)
from_elem!(f32_5, 5, f32)
from_elem!(f32_6, 6, f32)

from_elem!(f64_2, 2, f64)
from_elem!(f64_3, 3, f64)
from_elem!(f64_4, 4, f64)
from_elem!(f64_5, 5, f64)
from_elem!(f64_6, 6, f64)

macro_rules! from_elem_complex {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let size = num::pow(10u, $size);
            let elem = Complex::<$ty>::new(rand::random(), rand::random());

            b.iter(|| {
                vec::from_elem(size, elem)
            })
        }
    }
}

from_elem_complex!(c64_2, 2, f32)
from_elem_complex!(c64_3, 3, f32)
from_elem_complex!(c64_4, 4, f32)
from_elem_complex!(c64_5, 5, f32)
from_elem_complex!(c64_6, 6, f32)

from_elem_complex!(c128_2, 2, f64)
from_elem_complex!(c128_3, 3, f64)
from_elem_complex!(c128_4, 4, f64)
from_elem_complex!(c128_5, 5, f64)
from_elem_complex!(c128_6, 6, f64)
