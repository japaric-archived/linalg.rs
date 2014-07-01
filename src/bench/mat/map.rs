use num::Complex;
use rand::distributions::{IndependentSample,Range};
use std::{num,rand};

use mat;
use super::super::test::Bencher;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! map {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::<$ty>::new(num::zero(), num::one());
            let mut rng = rand::task_rng();
            let size = num::pow(10f64, $size).sqrt() as uint;
            let size = (size, size);

            let mut m = mat::rand(size, &between, &mut rng);

            b.iter(|| {
                m.map(|x| x.cos())
            })
        }
    }
}

map!(f32_2, 2u, f32)
map!(f32_3, 3u, f32)
map!(f32_4, 4u, f32)
map!(f32_5, 5u, f32)
map!(f32_6, 6u, f32)

map!(f64_2, 2u, f64)
map!(f64_3, 3u, f64)
map!(f64_4, 4u, f64)
map!(f64_5, 5u, f64)
map!(f64_6, 6u, f64)

macro_rules! map_complex {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::<$ty>::new(num::zero(), num::one());
            let mut rng = rand::task_rng();
            let size = num::pow(10f64, $size).sqrt() as uint;
            let size = (size, size);

            let mut m = mat::from_fn(size, |_, _| {
                Complex::new(between.ind_sample(&mut rng),
                           between.ind_sample(&mut rng))
            });

            b.iter(|| {
                m.map(|x| x.conj())
            })
        }
    }
}

map_complex!(c64_2, 2u, f32)
map_complex!(c64_3, 3u, f32)
map_complex!(c64_4, 4u, f32)
map_complex!(c64_5, 5u, f32)
map_complex!(c64_6, 6u, f32)

map_complex!(c128_2, 2u, f64)
map_complex!(c128_3, 3u, f64)
map_complex!(c128_4, 4u, f64)
map_complex!(c128_5, 5u, f64)
map_complex!(c128_6, 6u, f64)
