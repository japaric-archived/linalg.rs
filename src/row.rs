use {Col, Row};
use traits::{Collection, Matrix, Transpose};

impl<V> Matrix for Row<V> where V: Collection {
    fn ncols(&self) -> uint {
        Collection::len(&self.0)
    }

    fn nrows(&self) -> uint {
        1
    }
}

impl<V> Transpose<Col<V>> for Row<V> {
    fn t(self) -> Col<V> {
        Col(self.0)
    }
}
