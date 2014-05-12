use mat;
use num::complex::Cmplx;
use rand::random;
use std::num::pow;
use super::super::test::Bencher;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! from_elem {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let size = pow(10.0, $size).sqrt() as uint;
            let size = (size, size);
            let elem = random::<$ty>();

            b.iter(|| {
                mat::from_elem(size, elem)
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

macro_rules! from_elem_cmplx {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let size = pow(10.0, $size).sqrt() as uint;
            let size = (size, size);
            let elem = Cmplx::new(random::<$ty>(), random::<$ty>());

            b.iter(|| {
                mat::from_elem(size, elem)
            })
        }
    }
}

from_elem_cmplx!(c64_2, 2, f32)
from_elem_cmplx!(c64_3, 3, f32)
from_elem_cmplx!(c64_4, 4, f32)
from_elem_cmplx!(c64_5, 5, f32)
from_elem_cmplx!(c64_6, 6, f32)

from_elem_cmplx!(c128_2, 2, f64)
from_elem_cmplx!(c128_3, 3, f64)
from_elem_cmplx!(c128_4, 4, f64)
from_elem_cmplx!(c128_5, 5, f64)
from_elem_cmplx!(c128_6, 6, f64)
