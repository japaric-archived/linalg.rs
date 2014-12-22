use traits::{Matrix, Transpose};
use {Col, MutCol, MutRow, Row};

impl<'a, T> Matrix for MutRow<'a, T> {
    fn ncols(&self) -> uint {
        self.len()
    }

    fn nrows(&self) -> uint {
        1
    }
}

impl<'a, T> Transpose<MutCol<'a, T>> for MutRow<'a, T> {
    fn t(self) -> MutCol<'a, T> {
        MutCol(self.0)
    }
}

impl<'a, T> Matrix for Row<'a, T> {
    fn ncols(&self) -> uint {
        self.len()
    }

    fn nrows(&self) -> uint {
        1
    }
}

impl<'a, T> Transpose<Col<'a, T>> for Row<'a, T> {
    fn t(self) -> Col<'a, T> {
        Col(self.0)
    }
}
