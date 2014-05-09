use super::super::test::Bencher;
use vec;

macro_rules! clone {
    ($name:ident, $size:expr, $ty:ty) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let v = vec::ones::<$ty>($size);

            b.iter(|| {
                v.clone()
            })
        }
    }
}

clone!(f32_100, 100, f32)
clone!(f32_10_000, 10_000, f32)
clone!(f32_1_000_000, 1_000_000, f32)

clone!(f64_100, 100, f64)
clone!(f64_10_000, 10_000, f64)
clone!(f64_1_000_000, 1_000_000, f64)
