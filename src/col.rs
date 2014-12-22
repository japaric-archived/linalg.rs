use traits::{Matrix, Transpose};
use {Col, MutCol, MutRow, Row};

impl<'a, T> Matrix for MutCol<'a, T> {
    fn ncols(&self) -> uint {
        1
    }

    fn nrows(&self) -> uint {
        self.len()
    }
}

impl<'a, T> Transpose<MutRow<'a, T>> for MutCol<'a, T> {
    fn t(self) -> MutRow<'a, T> {
        MutRow(self.0)
    }
}

impl<'a, T> Matrix for Col<'a, T> {
    fn ncols(&self) -> uint {
        1
    }

    fn nrows(&self) -> uint {
        self.len()
    }
}

impl<'a, T> Transpose<Row<'a, T>> for Col<'a, T> {
    fn t(self) -> Row<'a, T> {
        Row(self.0)
    }
}
