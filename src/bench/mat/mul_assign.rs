use mat;
use rand::distributions::range::Range;
use std::rand::task_rng;
use std::num::pow;
use super::super::test::Bencher;
use traits::MulAssign;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! mul_assign {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::new(0 as $ty, 1 as $ty);
            let mut rng = task_rng();
            let size = pow(10f64, $size).sqrt() as uint;
            let size = (size, size);

            let mut x = mat::rand(size, &between, &mut rng);
            let y = mat::rand(size, &between, &mut rng);

            b.iter(|| {
                x.mul_assign(&y)
            })
        }
    }
}

mul_assign!(f32_2, 2u, f32)
mul_assign!(f32_3, 3u, f32)
mul_assign!(f32_4, 4u, f32)
mul_assign!(f32_5, 5u, f32)
mul_assign!(f32_6, 6u, f32)

mul_assign!(f64_2, 2u, f64)
mul_assign!(f64_3, 3u, f64)
mul_assign!(f64_4, 4u, f64)
mul_assign!(f64_5, 5u, f64)
mul_assign!(f64_6, 6u, f64)
