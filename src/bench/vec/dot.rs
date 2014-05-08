use array::traits::ArrayDot;
use super::super::test::Bencher;
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! dot {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let x = vec::ones::<$ty>($size);
            let y = vec::ones::<$ty>($size);

            b.iter(|| {
                x.dot(&y)
            })
        }
    }
}

dot!(fallback_100, 100, int)
dot!(fallback_10_000, 10_000, int)
dot!(fallback_1_000_000, 1_000_000, int)

dot!(sdot_100, 100, f32)
dot!(sdot_10_000, 10_000, f32)
dot!(sdot_1_000_000, 1_000_000, f32)

dot!(ddot_100, 100, f64)
dot!(ddot_10_000, 10_000, f64)
dot!(ddot_1_000_000, 1_000_000, f64)
