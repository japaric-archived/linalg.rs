use num::complex::Complex;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;
use rand::task_rng;
use std::num::pow;
use super::super::test::Bencher;
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! clone {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::new(0 as $ty, 1 as $ty);
            let mut rng = task_rng();
            let size = pow(10u, $size);

            let v = vec::rand(size, &between, &mut rng);

            b.iter(|| {
                v.clone()
            })
        }
    }
}

clone!(f32_2, 2, f32)
clone!(f32_3, 3, f32)
clone!(f32_4, 4, f32)
clone!(f32_5, 5, f32)
clone!(f32_6, 6, f32)

clone!(f64_2, 2, f64)
clone!(f64_3, 3, f64)
clone!(f64_4, 4, f64)
clone!(f64_5, 5, f64)
clone!(f64_6, 6, f64)

macro_rules! clone_complex {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::new(0 as $ty, 1 as $ty);
            let mut rng = task_rng();
            let size = pow(10u, $size);

            let v = vec::from_fn(size, |_| {
                Complex::new(between.ind_sample(&mut rng),
                           between.ind_sample(&mut rng))
            });

            b.iter(|| {
                v.clone()
            })
        }
    }
}

clone_complex!(c64_2, 2, f32)
clone_complex!(c64_3, 3, f32)
clone_complex!(c64_4, 4, f32)
clone_complex!(c64_5, 5, f32)
clone_complex!(c64_6, 6, f32)

clone_complex!(c128_2, 2, f64)
clone_complex!(c128_3, 3, f64)
clone_complex!(c128_4, 4, f64)
clone_complex!(c128_5, 5, f64)
clone_complex!(c128_6, 6, f64)
