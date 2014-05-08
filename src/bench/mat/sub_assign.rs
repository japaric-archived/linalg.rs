use num::complex::Cmplx;
use super::super::test::Bencher;
use traits::SubAssign;
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! sub_assign {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut x = vec::ones::<$ty>($size);
            let y = vec::ones::<$ty>($size);

            b.iter(|| {
                x.sub_assign(&y)
            })
        }
    }
}

sub_assign!(fallback_100, 100, int)
sub_assign!(fallback_10_000, 10_000, int)
sub_assign!(fallback_1_000_000, 1_000_000, int)

sub_assign!(saxpy_100, 100, f32)
sub_assign!(saxpy_10_000, 10_000, f32)
sub_assign!(saxpy_1_000_000, 1_000_000, f32)

sub_assign!(daxpy_100, 100, f64)
sub_assign!(daxpy_10_000, 10_000, f64)
sub_assign!(daxpy_1_000_000, 1_000_000, f64)

sub_assign!(caxpy_100, 100, Cmplx<f32>)
sub_assign!(caxpy_10_000, 10_000, Cmplx<f32>)
sub_assign!(caxpy_1_000_000, 1_000_000, Cmplx<f32>)

sub_assign!(zaxpy_100, 100, Cmplx<f64>)
sub_assign!(zaxpy_10_000, 10_000, Cmplx<f64>)
sub_assign!(zaxpy_1_000_000, 1_000_000, Cmplx<f64>)
