use mat;
use rand::distributions::range::Range;
use rand::task_rng;
use super::super::test::Bencher;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! from_elem {
    ($name:ident, $size:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            b.iter(|| {
                mat::from_elem(($size, $size), 1.1)
            })
        }
    }
}

from_elem!(from_elem_10, 10)
from_elem!(from_elem_100, 100)
from_elem!(from_elem_1_000, 1_000)

macro_rules! from_fn {
    ($name:ident, $size:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            b.iter(|| {
                mat::from_fn(($size, $size), |i, j| i + j)
            })
        }
    }
}

from_fn!(from_fn_10, 10)
from_fn!(from_fn_100, 100)
from_fn!(from_fn_1_000, 1_000)

macro_rules! rand {
    ($name:ident, $size:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let between = Range::new(0.0, 1.0);
            let mut rng = task_rng();

            b.iter(|| {
                mat::rand(($size, $size), &between, &mut rng)
            })
        }
    }
}

rand!(rand_10, 10)
rand!(rand_100, 100)
rand!(rand_1_000, 1_000)
