use num::complex::Cmplx;
use super::super::test::Bencher;
use traits::AddAssign;
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! add_assign {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut x = vec::ones::<$ty>($size);
            let y = vec::ones::<$ty>($size);

            b.iter(|| {
                x.add_assign(&y)
            })
        }
    }
}

add_assign!(fallback_100, 100, int)
add_assign!(fallback_10_000, 10_000, int)
add_assign!(fallback_1_000_000, 1_000_000, int)

add_assign!(saxpy_100, 100, f32)
add_assign!(saxpy_10_000, 10_000, f32)
add_assign!(saxpy_1_000_000, 1_000_000, f32)

add_assign!(daxpy_100, 100, f64)
add_assign!(daxpy_10_000, 10_000, f64)
add_assign!(daxpy_1_000_000, 1_000_000, f64)

add_assign!(caxpy_100, 100, Cmplx<f32>)
add_assign!(caxpy_10_000, 10_000, Cmplx<f32>)
add_assign!(caxpy_1_000_000, 1_000_000, Cmplx<f32>)

add_assign!(zaxpy_100, 100, Cmplx<f64>)
add_assign!(zaxpy_10_000, 10_000, Cmplx<f64>)
add_assign!(zaxpy_1_000_000, 1_000_000, Cmplx<f64>)
