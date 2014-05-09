use array::traits::ArrayShape;
use mat::{Col,Row};

pub trait MatrixCol {
    fn col(self, col: uint) -> Col<Self>;
    unsafe fn unsafe_col(self, col: uint) -> Col<Self>;
}

pub trait MatrixRow {
    fn row(self, row: uint) -> Row<Self>;
    unsafe fn unsafe_row(self, row: uint) -> Row<Self>;
}

pub trait MatrixShape: ArrayShape<(uint, uint)> {
    fn ncols(self) -> uint;
    fn nrows(self) -> uint;
}
