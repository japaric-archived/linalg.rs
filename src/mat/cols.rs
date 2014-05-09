use mat::Col;
use mat::traits::{MatrixCol,MatrixShape};
use std::mem::replace;

// TODO mozilla/rust#13302 Enforce Copy on M
pub struct Cols<M> {
    mat: M,
    state: uint,
    stop: uint,
}

impl<
    M: Copy + MatrixShape
> Cols<M> {
    pub fn new(mat: M) -> Cols<M> {
        Cols {
            mat: mat,
            state: 0,
            stop: mat.ncols(),
        }
    }
}

impl <
    M: Copy + MatrixCol
> Iterator<Col<M>>
for Cols<M> {
    fn next(&mut self) -> Option<Col<M>> {
        if self.state < self.stop {
            Some(unsafe {
                self.mat.unsafe_col(replace(&mut self.state, self.state + 1))
            })
        } else {
            None
        }
    }
}
