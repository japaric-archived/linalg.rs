use traits::{Matrix, Transpose};
use {Col, MutCol, MutRow, Row};

impl<'a, T> Matrix for MutCol<'a, T> {
    type Elem = T;

    fn ncols(&self) -> usize {
        1
    }

    fn nrows(&self) -> usize {
        self.len()
    }
}

impl<'a, T> Transpose for MutCol<'a, T> {
    type Output = MutRow<'a, T>;

    fn t(self) -> MutRow<'a, T> {
        MutRow(self.0)
    }
}

impl<'a, T> Matrix for Col<'a, T> {
    type Elem = T;

    fn ncols(&self) -> usize {
        1
    }

    fn nrows(&self) -> usize {
        self.len()
    }
}

impl<'a, T> Transpose for Col<'a, T> {
    type Output = Row<'a, T>;

    fn t(self) -> Row<'a, T> {
        Row(self.0)
    }
}
