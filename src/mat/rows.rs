use std::mem;

use mat::Row;
use mat::traits::{MatrixRow,MatrixShape};

// TODO mozilla/rust#13302 Enforce Copy on M
pub struct Rows<M> {
    mat: M,
    state: uint,
    stop: uint,
}

impl<
    M: Copy + MatrixShape
> Rows<M> {
    #[inline]
    pub fn new(mat: M) -> Rows<M> {
        Rows {
            mat: mat,
            state: 0,
            stop: mat.nrows(),
        }
    }
}

impl <
    M: Copy + MatrixRow
> Iterator<Row<M>>
for Rows<M> {
    #[inline]
    fn next(&mut self) -> Option<Row<M>> {
        let state = self.state;

        if state < self.stop {
            Some(unsafe {
                self.mat.unsafe_row(mem::replace(&mut self.state, state + 1))
            })
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        let exact = self.stop - self.state;

        (exact, Some(exact))
    }
}
