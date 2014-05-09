use mat;
use super::super::test::Bencher;

macro_rules! clone {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let m = mat::ones::<$ty>(($size, $size));

            b.iter(|| {
                m.clone()
            })
        }
    }
}

clone!(f32_10, 10, f32)
clone!(f32_100, 100, f32)
clone!(f32_1_000, 1_000, f32)

clone!(f64_10, 10, f64)
clone!(f64_100, 100, f64)
clone!(f64_1_000, 1_000, f64)
