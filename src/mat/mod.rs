use array::Array;
use rand::Rng;
use rand::distributions::IndependentSample;
use std::num::{One,Zero,one,zero};

pub type Mat<T> = Array<(uint, uint), T>;

#[inline]
pub fn from_elem<T: Clone>(shape: (uint, uint), elem: T) -> Mat<T> {
    let (nrows, ncols) = shape;

    unsafe {
        Array::from_raw_parts(Vec::from_elem(nrows * ncols, elem), shape)
    }
}

// TODO fork-join parallelism?
pub fn from_fn<T>(shape: (uint, uint), op: |uint, uint| -> T) -> Mat<T> {
    let (nrows, ncols) = shape;
    let mut v = Vec::new();

    for i in range(0, nrows) {
        for j in range(0, ncols) {
            v.push(op(i, j));
        }
    }

    unsafe { Array::from_raw_parts(v, shape) }
}

#[inline]
pub fn ones<T: Clone + One>(size: (uint, uint)) -> Mat<T> {
    from_elem(size, one())
}

#[inline]
pub fn rand<
    T,
    D: IndependentSample<T>,
    R: Rng
>(shape: (uint, uint), dist: &D, rng: &mut R) -> Mat<T> {
    let (nrows, ncols) = shape;

    unsafe {
        Array::from_raw_parts(
            range(0, nrows * ncols).map(|_| dist.ind_sample(rng)).collect(),
            shape
        )
    }
}

#[inline]
pub fn zeros<T: Clone + Zero>(size: (uint, uint)) -> Mat<T> {
    from_elem(size, zero())
}
