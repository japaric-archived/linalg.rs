use traits::{One, Zero};

macro_rules! float {
    ($($ty:ty),+) => {$(
        impl One for $ty {
            fn one() -> $ty {
                1.
            }
        }

        impl Zero for $ty {
            fn zero() -> $ty {
                0.
            }
        }
        )+
    }
}

float!(f32, f64)


macro_rules! int {
    ($($ty:ty),+) => {$(
        impl One for $ty {
            fn one() -> $ty {
                1
            }
        }

        impl Zero for $ty {
            fn zero() -> $ty {
                0
            }
        }
        )+
    }
}

int!(i8, i16, i32, i64, int, u8, u16, u32, u64, uint)
