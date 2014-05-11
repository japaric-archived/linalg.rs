use array::traits::ArrayShape;
use mat::{Col,Cols,Diag,Row,Rows};

pub trait MatrixCol {
    fn col(self, col: uint) -> Col<Self>;
    unsafe fn unsafe_col(self, col: uint) -> Col<Self>;
}

pub trait MatrixColIterator {
    fn cols(self) -> Cols<Self>;
}

pub trait MatrixDiag {
    fn diag(self, diag: int) -> Diag<Self>;
}

pub trait MatrixRow {
    fn row(self, row: uint) -> Row<Self>;
    unsafe fn unsafe_row(self, row: uint) -> Row<Self>;
}

pub trait MatrixRowIterator {
    fn rows(self) -> Rows<Self>;
}

pub trait MatrixShape: ArrayShape<(uint, uint)> {
    fn ncols(self) -> uint {
        let (_, ncols) = self.shape();

        ncols
    }

    fn nrows(self) -> uint {
        let (nrows, _) = self.shape();

        nrows
    }
}

pub trait MatrixView<V> {
    fn view(self, start: (uint, uint), stop: (uint, uint)) -> V;
    unsafe fn unsafe_view(self, start: (uint, uint), stop: (uint, uint)) -> V;
}
