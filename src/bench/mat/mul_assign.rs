use mat;
use super::super::test::Bencher;
use traits::MulAssign;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! mul_assign {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut x = mat::ones::<$ty>(($size, $size));
            let y = mat::ones::<$ty>(($size, $size));

            b.iter(|| {
                x.mul_assign(&y)
            })
        }
    }
}

mul_assign!(fallback_10, 10, int)
mul_assign!(fallback_100, 100, int)
mul_assign!(fallback_1_000, 1_000, int)

mul_assign!(f32x4_10, 10, f32)
mul_assign!(f32x4_100, 100, f32)
mul_assign!(f32x4_1_000, 1_000, f32)

mul_assign!(f64x2_10, 10, f64)
mul_assign!(f64x2_100, 100, f64)
mul_assign!(f64x2_1_000, 1_000, f64)
