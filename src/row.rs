use traits::{Matrix, Transpose};
use {Col, MutCol, MutRow, Row};

impl<'a, T> Matrix for MutRow<'a, T> {
    type Elem = T;

    fn ncols(&self) -> usize {
        self.len()
    }

    fn nrows(&self) -> usize {
        1
    }
}

impl<'a, T> Transpose for MutRow<'a, T> {
    type Output = MutCol<'a, T>;

    fn t(self) -> MutCol<'a, T> {
        MutCol(self.0)
    }
}

impl<'a, T> Matrix for Row<'a, T> {
    type Elem = T;

    fn ncols(&self) -> usize {
        self.len()
    }

    fn nrows(&self) -> usize {
        1
    }
}

impl<'a, T> Transpose for Row<'a, T> {
    type Output = Col<'a, T>;

    fn t(self) -> Col<'a, T> {
        Col(self.0)
    }
}
