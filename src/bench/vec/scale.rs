use array::traits::ArrayScale;
use num::complex::Cmplx;
use std::num::one;
use super::super::test::Bencher;
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! scale {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut x = vec::ones::<$ty>($size);

            b.iter(|| {
                x.scale(one::<$ty>() + one::<$ty>())
            })
        }
    }
}

scale!(fallback_100, 100, int)
scale!(fallback_10_000, 10_000, int)
scale!(fallback_1_000_000, 1_000_000, int)

scale!(sscal_100, 100, f32)
scale!(sscal_10_000, 10_000, f32)
scale!(sscal_1_000_000, 1_000_000, f32)

scale!(dscal_100, 100, f64)
scale!(dscal_10_000, 10_000, f64)
scale!(dscal_1_000_000, 1_000_000, f64)

scale!(cscal_100, 100, Cmplx<f32>)
scale!(cscal_10_000, 10_000, Cmplx<f32>)
scale!(cscal_1_000_000, 1_000_000, Cmplx<f32>)

scale!(zscal_100, 100, Cmplx<f64>)
scale!(zscal_10_000, 10_000, Cmplx<f64>)
scale!(zscal_1_000_000, 1_000_000, Cmplx<f64>)
