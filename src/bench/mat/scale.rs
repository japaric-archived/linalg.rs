use array::traits::ArrayScale;
use num::complex::Cmplx;
use std::num::one;
use super::super::test::Bencher;
use mat;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! scale {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut x = mat::ones::<$ty>(($size, $size));

            b.iter(|| {
                x.scale(one::<$ty>() + one::<$ty>())
            })
        }
    }
}

scale!(fallback_10, 10, int)
scale!(fallback_100, 100, int)
scale!(fallback_1_000, 1_000, int)

scale!(sscal_10, 10, f32)
scale!(sscal_100, 100, f32)
scale!(sscal_1_000, 1_000, f32)

scale!(dscal_10, 10, f64)
scale!(dscal_100, 100, f64)
scale!(dscal_1_000, 1_000, f64)

scale!(cscal_10, 10, Cmplx<f32>)
scale!(cscal_100, 100, Cmplx<f32>)
scale!(cscal_1_000, 1_000, Cmplx<f32>)

scale!(zscal_10, 10, Cmplx<f64>)
scale!(zscal_100, 100, Cmplx<f64>)
scale!(zscal_1_000, 1_000, Cmplx<f64>)
