use super::super::test::Bencher;
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! cos {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut v = vec::ones::<f32>($size);

            b.iter(|| {
                v.map(|x| x.cos())
            })
        }
    }
}

cos!(cos_100, 100, int)
cos!(cos_10_000, 10_000, int)
cos!(cos_1_000_000, 1_000_000, int)
