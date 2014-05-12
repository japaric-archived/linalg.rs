use num::complex::Cmplx;
use rand::distributions::IndependentSample;
use rand::distributions::range::Range;
use rand::task_rng;
use std::num::pow;
use super::super::test::Bencher;
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! map {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::new(0 as $ty, 1 as $ty);
            let mut rng = task_rng();
            let size = pow(10u, $size);

            let mut v = vec::rand(size, &between, &mut rng);

            b.iter(|| {
                v.map(|x| x.cos())
            })
        }
    }
}

map!(f32_2, 2, f32)
map!(f32_3, 3, f32)
map!(f32_4, 4, f32)
map!(f32_5, 5, f32)
map!(f32_6, 6, f32)

map!(f64_2, 2, f64)
map!(f64_3, 3, f64)
map!(f64_4, 4, f64)
map!(f64_5, 5, f64)
map!(f64_6, 6, f64)

macro_rules! map_cmplx {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::new(0 as $ty, 1 as $ty);
            let mut rng = task_rng();
            let size = pow(10u, $size);

            let mut v = vec::from_fn(size, |_| {
                Cmplx::new(between.ind_sample(&mut rng),
                           between.ind_sample(&mut rng))
            });

            b.iter(|| {
                v.map(|x| x.conj())
            })
        }
    }
}

map_cmplx!(c64_2, 2, f32)
map_cmplx!(c64_3, 3, f32)
map_cmplx!(c64_4, 4, f32)
map_cmplx!(c64_5, 5, f32)
map_cmplx!(c64_6, 6, f32)

map_cmplx!(c128_2, 2, f64)
map_cmplx!(c128_3, 3, f64)
map_cmplx!(c128_4, 4, f64)
map_cmplx!(c128_5, 5, f64)
map_cmplx!(c128_6, 6, f64)
