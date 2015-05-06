use blas::Gemm;
use cast::From;
use extract::Extract;
use lapack::{Getrf, Getri};
use onezero::{One, Zero};

use traits::{Eval, Matrix, MatrixInverse, SliceMut};
use {Chain, Mat, Scaled, SubMatMut, Transposed};

unsafe fn inv<T>(m: SubMatMut<T>) where T: Getri + Getrf {
    debug_assert_eq!(m.nrows(), m.ncols());

    let getri = T::getri();
    let getrf = T::getrf();

    let ref n = m.0.nrows;
    let mut ipiv = Vec::with_capacity(usize::from(*n).extract());
    let ipiv = ipiv.as_mut_ptr();

    let a = *m.0.data;
    let ref lda = m.0.stride;
    let ref mut info = 0;

    getrf(n, n, a, lda, ipiv, info);

    assert!(*info == 0);

    let lwork = n;
    let mut work = Vec::with_capacity(usize::from(*lwork).extract());
    let work = work.as_mut_ptr();

    getri(n, a, lda, ipiv, work, lwork, info);

    assert!(*info == 0);
}

// NOTE Core
impl<T> MatrixInverse for Mat<T> where T: Getri + Getrf {
    type Output = Mat<T>;

    fn inv(mut self) -> Mat<T> {
        unsafe {
            assert_eq!(self.nrows(), self.ncols());

            inv(self.slice_mut(..));

            self
        }
    }
}

// NOTE Secondary
impl<'a, T> MatrixInverse for Scaled<Chain<'a, T>> where T: Gemm + Getrf + Getri + One + Zero {
    type Output = Mat<T>;

    fn inv(self) -> Mat<T> {
        assert_eq!(self.nrows(), self.ncols());

        self.eval().inv()
    }
}

// NOTE Secondary
// Remember that (A^t)^-1 === (A^-1)^t
impl<'a, T> MatrixInverse for Transposed<Mat<T>> where T: Getrf + Getri {
    type Output = Transposed<Mat<T>>;

    fn inv(mut self) -> Transposed<Mat<T>> {
        unsafe {
            inv(self.0.slice_mut(..));

            self
        }
    }
}

// NOTE Forward
impl<'a, T> MatrixInverse for Chain<'a, T> where T: Gemm + Getrf + Getri + One + Zero {
    type Output = Mat<T>;

    fn inv(self) -> Mat<T> {
        Scaled(T::one(), self).inv()
    }
}
