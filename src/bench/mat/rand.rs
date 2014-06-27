use mat;
use rand::distributions::range::Range;
use std::rand::task_rng;
use std::num::pow;
use super::super::test::Bencher;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! rand {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::new(0 as $ty, 1 as $ty);
            let mut rng = task_rng();
            let size = pow(10f64, $size).sqrt() as uint;
            let size = (size, size);

            b.iter(|| {
                mat::rand(size, &between, &mut rng)
            })
        }
    }
}

rand!(f32_2, 2u, f32)
rand!(f32_3, 3u, f32)
rand!(f32_4, 4u, f32)
rand!(f32_5, 5u, f32)
rand!(f32_6, 6u, f32)

rand!(f64_2, 2u, f64)
rand!(f64_3, 3u, f64)
rand!(f64_4, 4u, f64)
rand!(f64_5, 5u, f64)
rand!(f64_6, 6u, f64)
