use std::{mem, slice};
use std::num::{One, Zero, mod};

use Mat;
use blas::gemm::BlasGemm;
use blas::{BLAS_NO_TRANS, to_blasint};
use notsafe::{UnsafeIndex, UnsafeIndexMut};
use traits::{Collection, Iter, Matrix, MutIter};

mod col;
mod cols;
mod diag;
mod mutcol;
mod mutcols;
mod mutdiag;
mod mutrow;
mod mutrows;
mod mutview;
mod row;
mod rows;
mod trans;
mod view;

impl<T> Collection for Mat<T> {
    fn len(&self) -> uint {
        self.data.len()
    }
}

impl<'a, T> Iter<'a, &'a T, slice::Items<'a, T>> for Mat<T> {
    fn iter(&'a self) -> slice::Items<'a, T> {
        self.data.iter()
    }
}

impl<D> Matrix for Mat<D> {
    fn ncols(&self) -> uint {
        self.data.len() / self.stride
    }

    fn nrows(&self) -> uint {
        self.stride
    }
}

impl<T> Mul<Mat<T>, Mat<T>> for Mat<T> where T: BlasGemm + One + Zero {
    /// - Memory: `O(lhs.nrows * rhs.ncols)`
    /// - Time: `O(lhs.nrows * lhs.ncols * rhs.ncols)`
    ///
    /// # Panics
    ///
    /// Panics if `lhs.ncols != rhs.nrows`
    fn mul(&self, rhs: &Mat<T>) -> Mat<T> {
        assert!(self.ncols() == rhs.nrows());

        let length = self.nrows().checked_mul(&rhs.ncols()).unwrap();
        let mut data = Vec::with_capacity(length);
        unsafe { data.set_len(length) }

        let gemm = BlasGemm::gemm(None::<T>);
        let transa = &BLAS_NO_TRANS;
        let transb = &BLAS_NO_TRANS;
        let m = &to_blasint(self.nrows());
        let n = &to_blasint(rhs.ncols());
        let k = &to_blasint(self.ncols());
        let alpha = &num::one();
        let a = self.data.as_ptr();
        let lda = m;
        let b = rhs.data.as_ptr();
        let ldb = k;
        let beta = &num::zero();
        let c = data.as_mut_ptr();
        let ldc = m;
        unsafe { gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }

        Mat::new(data, self.nrows())
    }
}

impl<'a, T> MutIter<'a, &'a mut T, slice::MutItems<'a, T>> for Mat<T> {
    fn mut_iter(&'a mut self) -> slice::MutItems<'a, T> {
        self.data.iter_mut()
    }
}

impl<T> UnsafeIndex<(uint, uint), T> for Mat<T> {
    unsafe fn unsafe_index(&self, &(row, col): &(uint, uint)) -> &T {
        mem::transmute(self.data.as_ptr().offset((col * self.stride + row) as int))
    }
}

impl<T> UnsafeIndexMut<(uint, uint), T> for Mat<T> {
    unsafe fn unsafe_index_mut(&mut self, &(row, col): &(uint, uint)) -> &mut T {
        mem::transmute(self.data.as_mut_ptr().offset((col * self.stride + row) as int))
    }
}

#[cfg(test)]
mod test {
    use quickcheck::TestResult;
    use std::collections::TreeSet;

    use test;
    use traits::{Collection, Iter, MutIter, OptionIndex, OptionIndexMut};

    #[quickcheck]
    fn index(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        if let Some(e) = test::mat(size).as_ref().and_then(|m| m.at(&(row, col))) {
            TestResult::from_bool((row, col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn index_mut(size: (uint, uint), (row, col): (uint, uint)) -> TestResult {
        if let Some(e) = test::mat(size).as_mut().and_then(|m| m.at_mut(&(row, col))) {
            TestResult::from_bool((row, col).eq(e))
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn iter(size: (uint, uint)) -> TestResult {
        if let Some(m) = test::mat(size) {
            let (nrows, ncols) = size;

            let mut elems = TreeSet::new();
            for row in range(0, nrows) {
                for col in range(0, ncols) {
                    elems.insert((row, col));
                }
            }

            TestResult::from_bool(elems == m.iter().map(|&x| x).collect())
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_iter(size: (uint, uint)) -> TestResult {
        if let Some(mut m) = test::mat(size) {
            let (nrows, ncols) = size;

            let mut elems = TreeSet::new();
            for row in range(0, nrows) {
                for col in range(0, ncols) {
                    elems.insert((row, col));
                }
            }

            TestResult::from_bool(elems == m.mut_iter().map(|&x| x).collect())
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn mut_size_hint(size: (uint, uint), skip: uint) -> TestResult {
        if let Some(mut m) = test::mat(size) {
            let total = m.len();

            if skip < total {
                let hint = m.mut_iter().skip(skip).size_hint();
                let left = total - skip;

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }

    #[quickcheck]
    fn size_hint(size: (uint, uint), skip: uint) -> TestResult {
        if let Some(m) = test::mat(size) {
            let total = m.len();

            if skip < total {
                let hint = m.iter().skip(skip).size_hint();
                let left = total - skip;

                TestResult::from_bool(hint == (left, Some(left)))
            } else {
                TestResult::discard()
            }
        } else {
            TestResult::discard()
        }
    }

    macro_rules! blas {
        ($($ty:ident),+) => {$(
            mod $ty {
                use quickcheck::TestResult;
                use std::iter::AdditiveIterator;

                use test;
                use traits::{Iter, MatrixCols, MatrixCol, MatrixRow, MatrixRows};

                #[quickcheck]
                fn mul(m: uint, k: uint, n: uint) -> TestResult {

                    if let (Some(x), Some(y)) = (
                        test::rand_mat::<$ty>((m, k)),
                        test::rand_mat::<$ty>((k, n)),
                    ) {
                        let z = x * y;

                        for (r, rx) in z.rows().zip(x.rows()) {
                            for (&e, cy) in r.iter().zip(y.cols()) {
                                let sum = rx.iter().zip(cy.iter()).map(|(&x, &y)| x * y).sum();

                                if !test::is_close(e, sum) {
                                    return TestResult::failed();
                                }
                            }
                        }

                        TestResult::passed()
                    } else {
                        TestResult::discard()
                    }
                }

                #[quickcheck]
                fn row_mul_col(length: uint, (row, col): (uint, uint)) -> TestResult {
                    if let Some((r, c)) =
                        test::rand_mat::<$ty>((length, length)).
                            as_ref().
                            and_then(|m| m.row(row).and_then(|r| m.col(col).map(|c| (r, c))))
                    {
                        TestResult::from_bool(test::is_close(
                            r * c,
                            r.iter().zip(c.iter()).map(|(x, y)| x.mul(y)).sum(),
                        ))
                    } else {
                        TestResult::discard()
                    }
                }
            }
        )+}
    }

    blas!(f32, f64)
}
