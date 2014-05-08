use array::Array;
use rand::Rng;
use rand::distributions::IndependentSample;
use std::num::{One,Zero,one,zero};

// XXX ugly name, but both Vec and Vector are taken :-(
pub type Vect<T> = Array<(uint,), T>;

#[inline]
pub fn from_elem<T: Clone>(size: uint, elem: T) -> Vect<T> {
    unsafe { Array::from_raw_parts(Vec::from_elem(size, elem), (size,)) }
}

// TODO fork-join parallelism?
#[inline]
pub fn from_fn<T>(size: uint, op: |uint| -> T) -> Vect<T> {
    unsafe { Array::from_raw_parts(Vec::from_fn(size, op), (size,)) }
}

#[inline]
pub fn ones<T: Clone + One>(size: uint) -> Vect<T> {
    from_elem(size, one())
}

#[inline]
pub fn rand<
    T,
    D: IndependentSample<T>,
    R: Rng
>(size: uint, dist: &D, rng: &mut R) -> Vect<T> {
    unsafe {
        Array::from_raw_parts(
            range(0, size).map(|_| dist.ind_sample(rng)).collect(),
            (size,)
        )
    }
}

#[inline]
pub fn zeros<T: Clone + Zero>(size: uint) -> Vect<T> {
    from_elem(size, zero())
}
