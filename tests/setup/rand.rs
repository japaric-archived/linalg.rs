use linalg::prelude::*;
use rand::{Rand, Rng, XorShiftRng, self};

pub fn col<T>(length: u32) -> ColVec<T> where T: Rand {
    let ref mut rng: XorShiftRng = rand::thread_rng().gen();

    (0..length).map(|_| rng.gen()).collect()
}

pub fn mat<T>(size: (u32, u32)) -> Mat<T> where T: Rand {
    let ref mut rng: XorShiftRng = rand::thread_rng().gen();

    Mat::from_fn(size, |_| rng.gen())
}

pub fn row<T>(length: u32) -> RowVec<T> where T: Rand {
    let ref mut rng: XorShiftRng = rand::thread_rng().gen();

    (0..length).map(|_| rng.gen()).collect()
}

pub fn scalar<T>() -> T where T: Rand {
    let ref mut rng: XorShiftRng = rand::thread_rng().gen();
    rng.gen()
}
