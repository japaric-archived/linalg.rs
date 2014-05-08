// TODO norm2
use array::traits::ArrayNorm2;
use num::complex::Cmplx;
use super::super::test::Bencher;
use vec;

// FIXME mozilla/rust#12249 DRYer benchmarks using macros
macro_rules! norm2 {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let x = vec::ones::<$ty>($size);

            b.iter(|| {
                x.norm2()
            })
        }
    }
}

norm2!(snrm2_100, 100, f32)
norm2!(snrm2_10_000, 10_000, f32)
norm2!(snrm2_1_000_000, 1_000_000, f32)

norm2!(dnrm2_100, 100, f64)
norm2!(dnrm2_10_000, 10_000, f64)
norm2!(dnrm2_1_000_000, 1_000_000, f64)

norm2!(scnrm2_100, 100, Cmplx<f32>)
norm2!(scnrm2_10_000, 10_000, Cmplx<f32>)
norm2!(scnrm2_1_000_000, 1_000_000, Cmplx<f32>)

norm2!(dznrm2_100, 100, Cmplx<f64>)
norm2!(dznrm2_10_000, 10_000, Cmplx<f64>)
norm2!(dznrm2_1_000_000, 1_000_000, Cmplx<f64>)
