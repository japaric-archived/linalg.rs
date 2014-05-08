use mat;
use super::super::test::Bencher;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! cos {
    ($name:ident, $size:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut m = mat::zeros::<f32>(($size, $size));

            b.iter(|| {
                m.map(|x| x.cos())
            })
        }
    }
}

cos!(cos_10, 10)
cos!(cos_100, 100)
cos!(cos_1_000, 1_000)
