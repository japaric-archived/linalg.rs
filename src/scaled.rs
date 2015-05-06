use traits::{Matrix, Transpose};
use Scaled;

impl<M> Matrix for Scaled<M> where M: Matrix {
    type Elem = M::Elem;

    fn ncols(&self) -> u32 {
        self.1.ncols()
    }

    fn nrows(&self) -> u32 {
        self.1.nrows()
    }

    fn size(&self) -> (u32, u32) {
        self.1.size()
    }
}

impl<T, M> Transpose for Scaled<M> where
    M: Matrix<Elem=T> + Transpose,
    M::Output: Matrix<Elem=T>,
{
    type Output = Scaled<M::Output>;

    fn t(self) -> Scaled<M::Output> {
        Scaled(self.0, self.1.t())
    }
}
