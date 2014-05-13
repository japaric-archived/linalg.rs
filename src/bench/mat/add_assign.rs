use mat;
use num::complex::Complex;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;
use rand::task_rng;
use std::num::pow;
use super::super::test::Bencher;
use traits::AddAssign;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! add_assign {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::new(0 as $ty, 1 as $ty);
            let mut rng = task_rng();
            let size = pow(10.0, $size).sqrt() as uint;
            let size = (size, size);

            let mut x = mat::rand(size, &between, &mut rng);
            let y = mat::rand(size, &between, &mut rng);

            b.iter(|| {
                x.add_assign(&y)
            })
        }
    }
}

add_assign!(f32_2, 2, f32)
add_assign!(f32_3, 3, f32)
add_assign!(f32_4, 4, f32)
add_assign!(f32_5, 5, f32)
add_assign!(f32_6, 6, f32)

add_assign!(f64_2, 2, f64)
add_assign!(f64_3, 3, f64)
add_assign!(f64_4, 4, f64)
add_assign!(f64_5, 5, f64)
add_assign!(f64_6, 6, f64)

macro_rules! add_assign_complex {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::new(0 as $ty, 1 as $ty);
            let mut rng = task_rng();
            let size = pow(10.0, $size).sqrt() as uint;
            let size = (size, size);

            let mut x = mat::from_fn(size, |_, _| {
                Complex::new(between.ind_sample(&mut rng),
                           between.ind_sample(&mut rng))
            });
            let y = mat::from_fn(size, |_, _| {
                Complex::new(between.ind_sample(&mut rng),
                           between.ind_sample(&mut rng))
            });

            b.iter(|| {
                x.add_assign(&y)
            })
        }
    }
}

add_assign_complex!(c64_2, 2, f32)
add_assign_complex!(c64_3, 3, f32)
add_assign_complex!(c64_4, 4, f32)
add_assign_complex!(c64_5, 5, f32)
add_assign_complex!(c64_6, 6, f32)

add_assign_complex!(c128_2, 2, f64)
add_assign_complex!(c128_3, 3, f64)
add_assign_complex!(c128_4, 4, f64)
add_assign_complex!(c128_5, 5, f64)
add_assign_complex!(c128_6, 6, f64)
