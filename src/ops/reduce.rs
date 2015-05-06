use std::ops::{Range, RangeFrom, RangeTo};

use blas::{Gemm, Gemv, Transpose};
use cast::From;
use extract::Extract;
use onezero::{One, Zero};

use ops::Reduce;
use ops::mcop::{Table, self};
use ops;
use traits::{Matrix, Slice};
use {Chain, Col, ColVec, Mat, Product, Row, RowVec, SubMat};

#[allow(non_camel_case_types)]
pub enum MatMulMat<'a, T> {
    M_M(Mat<T>, Mat<T>),
    M_SM(Mat<T>, (Transpose, SubMat<'a, T>)),
    SM_M((Transpose, SubMat<'a, T>), Mat<T>),
    SM_SM((Transpose, SubMat<'a, T>), (Transpose, SubMat<'a, T>))
}

impl<'a, T> Reduce for Chain<'a, T> where T: Gemm + One + Zero {
    type Output = MatMulMat<'a, T>;

    fn reduce(self) -> MatMulMat<'a, T> {
        unsafe {
            use self::MatMulMat::*;

            if self.tail.is_empty() {
                return SM_SM(self.first, self.second)
            }

            let n = self.len();
            let mut dims = Vec::with_capacity(n + 1);

            dims.push(u64::from(self.first.1.nrows()));
            dims.push(u64::from(self.first.1.ncols()));
            dims.push(u64::from(self.second.1.ncols()));

            for &(_, mat) in &self.tail {
                dims.push(u64::from(mat.ncols()));
            }

            let ref split = mcop::solve(&dims);
            let i = split[0..n];

            if i == 1 {
                SM_M(self.first, reduce_to_mat(&self, i..n, split))
            } else if i == n - 1 {
                M_SM(reduce_to_mat(&self, 0..i, split), *self.tail.get(i - 2).extract())
            } else {
                M_M(reduce_to_mat(&self, 0..i, split), reduce_to_mat(&self, i..n, split))
            }
        }
    }
}

#[allow(non_camel_case_types)]
pub enum MatMulCol<'a, 'b, T> {
    M_C(Mat<T>, Col<'b, T>),
    M_CV(Mat<T>, ColVec<T>),
    SM_CV((Transpose, SubMat<'a, T>), ColVec<T>),
}

impl<'a, 'b, T> Reduce for Product<Chain<'a, T>, Col<'b, T>> where
    T: Gemm + Gemv + One + Zero,
{
    type Output = MatMulCol<'a, 'b, T>;

    fn reduce(self) -> MatMulCol<'a, 'b, T> {
        unsafe {
            use self::MatMulCol::*;

            let Product(ref chain, col) = self;

            let n = chain.len() + 1;
            let mut dims = Vec::with_capacity(n + 1);

            dims.push(u64::from(chain.first.1.nrows()));
            dims.push(u64::from(chain.first.1.ncols()));
            dims.push(u64::from(chain.second.1.ncols()));

            for &(_, mat) in &chain.tail {
                dims.push(u64::from(mat.ncols()));
            }

            dims.push(1);

            let ref split = mcop::solve(&dims);
            let i = split[0..n];

            if i == 1 {
                SM_CV(chain.first, reduce_to_col(chain, col, i.., split))
            } else if i == n - 1 {
                M_C(reduce_to_mat(chain, 0..i, split), col)
            } else {
                M_CV(reduce_to_mat(chain, 0..i, split), reduce_to_col(chain, col, i.., split))
            }
        }
    }
}

#[allow(non_camel_case_types)]
pub enum RowMulCol<'a, 'b, T> {
    RV_C(RowVec<T>, Col<'b, T>),
    RV_CV(RowVec<T>, ColVec<T>),
    R_CV(Row<'a, T>, ColVec<T>),
}

impl<'a, 'b, 'c, T> Reduce for (Row<'a, T>, Chain<'b, T>, Col<'c, T>) where
    T: Gemm + Gemv + One + Zero,
{
    type Output = RowMulCol<'a, 'c, T>;

    fn reduce(self) -> RowMulCol<'a, 'c, T> {
        unsafe fn reduce_to_col<T>(
            chain: &Chain<T>,
            col: Col<T>,
            RangeFrom { start }: RangeFrom<usize>,
            split: &Table<usize>,
        ) -> ColVec<T> where
            T: Gemm + Gemv + One + Zero,
        {
            let end = chain.len() + 2;

            debug_assert!(start + 1 < end);
            debug_assert!(start >= 1);

            let i = split[start..end];
            let ref alpha = T::one();

            match (i == start + 1, i == end - 1) {
                (false, false) => {
                    let a = reduce_to_mat(chain, start..i, split);
                    let ref transa = Transpose::No;
                    let x = reduce_to_col(chain, col, i.., split);

                    ops::submat_mul_col(transa, alpha, a.slice(..), x.slice(..))
                },
                (false, true) => {
                    let a = reduce_to_mat(chain, start..i, split);
                    let ref transa = Transpose::No;
                    let x = col;

                    ops::submat_mul_col(transa, alpha, a.slice(..), x)
                },
                (true, false) => {
                    let (ref transa, a) = if start == 0 {
                        // row
                        None.extract()
                    } else if start == 1 {
                        chain.first
                    } else if start == 2 {
                        chain.second
                    } else {
                        *chain.tail.get(start - 3).extract()
                    };

                    let x = reduce_to_col(chain, col, i.., split);

                    ops::submat_mul_col(transa, alpha, a, x.slice(..))
                },
                (true, true) => {
                    let x = col;

                    if start == 0 {
                        // row * chain.first
                        None.extract()
                    } else if start == 1 {
                        // chain.first * chain.second
                        None.extract()
                    } else if start == 2 {
                        let (ref transa, a) = chain.second;

                        ops::submat_mul_col(transa, alpha, a, x)
                    } else {
                        let &(ref transa, a) = chain.tail.get(start - 3).extract();

                        ops::submat_mul_col(transa, alpha, a, x)
                    }
                },
            }
        }

        unsafe fn reduce_to_mat<T>(
            chain: &Chain<T>,
            Range{ start, end }: Range<usize>,
            split: &Table<usize>,
        ) -> Mat<T> where
            T: Gemm + Gemv + One + Zero,
        {
            debug_assert!(end <= chain.len() + 1);
            debug_assert!(start + 1 < end);
            debug_assert!(start >= 1);

            let i = split[start..end];
            let ref alpha = T::one();

            match (i == start + 1, i == end - 1) {
                (false, false) => {
                    let a = reduce_to_mat(chain, start..i, split);
                    let b = reduce_to_mat(chain, i..end, split);
                    let ref transa = Transpose::No;
                    let ref transb = Transpose::No;

                    ops::submat_mul_submat(transa, transb, alpha, a.slice(..), b.slice(..))
                },
                (false, true) => {
                    let a = reduce_to_mat(chain, start..i, split);
                    let ref transa = Transpose::No;

                    let (ref transb, b) = *chain.tail.get(i - 3).extract();

                    ops::submat_mul_submat(transa, transb, alpha, a.slice(..), b)
                },
                (true, false) => {
                    let b = reduce_to_mat(chain, i..end, split);
                    let ref transb = Transpose::No;

                    let (ref transa, a) = if start == 1 {
                        chain.first
                    } else if start == 2 {
                        chain.second
                    } else {
                        *chain.tail.get(start - 3).extract()
                    };

                    ops::submat_mul_submat(transa, transb, alpha, a, b.slice(..))
                },
                (true, true) => {
                    let ((ref transa, a), (ref transb, b)) = if start == 1 {
                        (chain.first, chain.second)
                    } else if start == 2 {
                        (chain.second, *chain.tail.first().extract())
                    } else {
                        (
                            *chain.tail.get(start - 3).extract(),
                            *chain.tail.get(start - 2).extract(),
                        )
                    };

                    ops::submat_mul_submat(transa, transb, alpha, a, b)
                },
            }
        }

        unsafe fn reduce_to_row<T>(
            row: Row<T>,
            chain: &Chain<T>,
            RangeTo { end }: RangeTo<usize>,
            split: &Table<usize>,
        ) -> RowVec<T> where
            T: Gemm + Gemv + One + Zero,
        {
            let start = 0;

            debug_assert!(end <= chain.len() + 1);
            debug_assert!(start + 1 < end);

            let i = split[start..end];
            let ref alpha = T::one();

            match (i == start + 1, i == end - 1) {
                (false, false) => {
                    let x = reduce_to_row(row, chain, ..i, split);
                    let a = reduce_to_mat(chain, i..end, split);
                    let ref transa = Transpose::No;

                    ops::row_mul_submat(transa, alpha, a.slice(..), x.slice(..))
                },
                (false, true) => {
                    let x = reduce_to_row(row, chain, ..i, split);

                    let (ref transa, a) = if i == 0 {
                        // row
                        None.extract()
                    } else if i == 1 {
                        chain.first
                    } else if i == 2 {
                        chain.second
                    } else {
                        *chain.tail.get(i - 3).extract()
                    };

                    ops::row_mul_submat(transa, alpha, a, x.slice(..))
                },
                (true, false) => {
                    let x = row;
                    let a = reduce_to_mat(chain, i..end, split);
                    let ref transa = Transpose::No;

                    ops::row_mul_submat(transa, alpha, a.slice(..), x)
                },
                (true, true) => {
                    let x = row;
                    let (ref transa, a) = chain.first;

                    ops::row_mul_submat(transa, alpha, a, x)
                },
            }
        }

        unsafe {
            use self::RowMulCol::*;

            let (row, chain, col) = self;
            let ref chain = chain;

            let n = chain.len() + 2;
            let mut dims = Vec::with_capacity(n + 1);

            dims.push(1);

            dims.push(u64::from(chain.first.1.nrows()));
            dims.push(u64::from(chain.first.1.ncols()));
            dims.push(u64::from(chain.second.1.ncols()));

            for &(_, mat) in &chain.tail {
                dims.push(u64::from(mat.ncols()));
            }

            dims.push(1);

            let ref split = mcop::solve(&dims);
            let i = split[0..n];

            if i == 1 {
                R_CV(row, reduce_to_col(chain, col, i.., split))
            } else if i == n - 1 {
                RV_C(reduce_to_row(row, chain, ..i, split), col)
            } else {
                RV_CV(reduce_to_row(row, chain, ..i, split), reduce_to_col(chain, col, i.., split))
            }
        }
    }
}

unsafe fn reduce_to_col<T>(
    chain: &Chain<T>,
    col: Col<T>,
    RangeFrom { start }: RangeFrom<usize>,
    split: &Table<usize>,
) -> ColVec<T> where
    T: Gemm + Gemv + One + Zero,
{
    let end = chain.len() + 1;
    let i = split[start..end];
    let ref alpha = T::one();

    debug_assert!(start + 1 < end);

    match (i == start + 1, i == end -1) {
        (false, false) => {
            let a = reduce_to_mat(chain, start..i, split);
            let ref transa = Transpose::No;
            let x = reduce_to_col(chain, col, i.., split);

            ops::submat_mul_col(transa, alpha, a.slice(..), x.slice(..))
        },
        (false, true) => {
            let a = reduce_to_mat(chain, start..i, split);
            let ref transa = Transpose::No;
            let x = col;

            ops::submat_mul_col(transa, alpha, a.slice(..), x)
        },
        (true, false) => {
            let (ref transa, a) = if start == 0 {
                chain.first
            } else if start == 1 {
                chain.second
            } else {
                *chain.tail.get(start - 2).extract()
            };

            let x = reduce_to_col(chain, col, i.., split);

            ops::submat_mul_col(transa, alpha, a, x.slice(..))
        },
        (true, true) => {
            let x = col;

            if start == 0 {
                // chain.first * chain.second
                None.extract()
            } else if start == 1 {
                let (ref transa, a) = chain.second;

                ops::submat_mul_col(transa, alpha, a, x)
            } else {
                let &(ref transa, a) = chain.tail.get(start - 2).extract();

                ops::submat_mul_col(transa, alpha, a, x)
            }
        },
    }
}

unsafe fn reduce_to_mat<T>(
    chain: &Chain<T>,
    Range { start, end }: Range<usize>,
    split: &Table<usize>,
) -> Mat<T> where
    T: Gemm + One + Zero,
{
    debug_assert!(end <= chain.len());
    debug_assert!(start + 1 < end);

    let ref alpha = T::one();
    let i = split[start..end];

    match (i == start + 1, i == end - 1) {
        (false, false) => {
            let a = reduce_to_mat(chain, start..i, split);
            let b = reduce_to_mat(chain, i..end, split);
            let ref transa = Transpose::No;
            let ref transb = Transpose::No;

            ops::submat_mul_submat(transa, transb, alpha, a.slice(..), b.slice(..))
        },
        (false, true) => {
            let a = reduce_to_mat(chain, start..i, split);
            let ref transa = Transpose::No;

            let (ref transb, b) = *chain.tail.get(i - 2).extract();

            ops::submat_mul_submat(transa, transb, alpha, a.slice(..), b)
        },
        (true, false) => {
            let b = reduce_to_mat(chain, i..end, split);
            let ref transb = Transpose::No;

            let (ref transa, a) = if start == 0 {
                chain.first
            } else if start == 1 {
                chain.second
            } else {
                *chain.tail.get(start - 2).extract()
            };

            ops::submat_mul_submat(transa, transb, alpha, a, b.slice(..))
        },
        (true, true) => {
            let ((ref transa, a), (ref transb, b)) = if start == 0 {
                (chain.first, chain.second)
            } else if start == 1 {
                (chain.second, *chain.tail.first().extract())
            } else {
                (*chain.tail.get(start - 2).extract(), *chain.tail.get(start - 1).extract())
            };

            ops::submat_mul_submat(transa, transb, alpha, a, b)
        },
    }
}
