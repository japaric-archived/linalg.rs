use Product;
use traits::{Matrix, Transpose};

impl<T, L, R> Matrix for Product<L, R> where L: Matrix<Elem=T>, R: Matrix<Elem=T> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        self.0.nrows()
    }

    fn ncols(&self) -> u32 {
        self.1.ncols()
    }
}

impl<L, R> Transpose for Product<L, R> where L: Transpose, R: Transpose {
    type Output = Product<R::Output, L::Output>;

    fn t(self) -> Product<R::Output, L::Output> {
        Product(self.1.t(), self.0.t())
    }
}
