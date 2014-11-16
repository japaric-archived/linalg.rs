use traits::{Collection, Matrix, Transpose};
use {Col, Row};

impl<V> Matrix for Col<V> where V: Collection {
    fn ncols(&self) -> uint {
        1
    }

    fn nrows(&self) -> uint {
        self.0.len()
    }
}

impl<V> Transpose<Row<V>> for Col<V> {
    fn t(self) -> Row<V> {
        Row(self.0)
    }
}
