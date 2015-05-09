use blas::{Axpy, Copy, Gemm, Gemv, Scal, Transpose};
use cast::From;
use extract::Extract;
use onezero::Zero;

use traits::Transpose as _0;
use traits::{Matrix, SliceMut};
use {Col, ColMut, ColVec, Mat, Row, RowVec, Slice, SubMat, SubMatMut, Tor};

macro_rules! assert_eq_inner_dimensions {
    ($lhs:expr, $rhs:expr) => {
        assert!($lhs.ncols() == $rhs.nrows() && $lhs.ncols() != 0);
    }
}

macro_rules! assert_eq_size {
    ($lhs:expr, $rhs:expr) => {
        assert!($lhs.size() == $rhs.size());
    }
}

mod add;
mod add_assign;
mod div_assign;
mod eq;
mod eval;
mod from;
mod inv;
mod mcop;
mod mul;
mod mul_assign;
mod norm;
mod product;
mod reduce;
mod scaled;
mod set;
mod sub;
mod sub_assign;

trait Reduce {
    type Output;

    fn reduce(self) -> Self::Output;
}

/// y := alpha * x + y
fn axpy_slice_scalar<T>(alpha: &T, x: &T, y: &mut [T]) where T: Axpy {
    unsafe {
        let axpy = T::axpy();
        let incx = &0;
        let incy = &1;

        let mut n = y.len();
        let mut y = y.as_mut_ptr();

        let max = usize::from(i32::max_value()).extract();
        let offset = isize::from(i32::max_value());

        while n >= max {
            axpy(&i32::max_value(), alpha, x, incx, y, incy);

            y = y.offset(offset);
            n -= max;
        }

        axpy(&i32::from(n).extract(), alpha, x, incx, y, incy)
    }
}

/// y := alpha * x + y
unsafe fn axpy_slice_slice<T>(alpha: &T, x: &[T], y: &mut [T]) where T: Axpy {
    debug_assert_eq!(y.len(), x.len());

    let axpy = T::axpy();
    let x = x.as_ptr();
    let incx = &1;
    let incy = &1;

    let mut n = y.len();
    let mut y = y.as_mut_ptr();

    let max = usize::from(i32::max_value()).extract();
    let offset = isize::from(i32::max_value());

    while n >= max {
        axpy(&i32::max_value(), alpha, x, incx, y, incy);

        y = y.offset(offset);
        n -= max;
    }

    axpy(&i32::from(n).extract(), alpha, x, incx, y, incy)
}

/// y := alpha * x + y
fn axpy_strided_scalar<T>(alpha: &T, x: &T, y: &mut Slice<T>) where T: Axpy {
    unsafe {
        let axpy = T::axpy();
        let n = &y.len;
        let incx = &0;
        let incy = &*y.stride;

        let y = *y.data;

        axpy(n, alpha, x, incx, y, incy)
    }
}

/// y := alpha * x + y
unsafe fn axpy_strided_strided<T>(alpha: &T, x: &Slice<T>, y: &mut Slice<T>) where T: Axpy {
    debug_assert_eq!(x.len, y.len);

    let axpy = T::axpy();
    let n = &y.len;
    let incx = &*x.stride;
    let incy = &*y.stride;

    let y = *y.data;
    let x = *x.data;

    axpy(n, alpha, x, incx, y, incy)
}

/// y := x
unsafe fn copy_strided<T>(input: &Slice<T>, output: &mut Slice<T>) where T: Copy {
    debug_assert_eq!(input.len, output.len);

    let copy = T::copy();
    let n = &input.len;
    let x = *input.data;
    let incx = &*input.stride;
    let y = *output.data;
    let incy = &*output.stride;

    copy(n, x, incx, y, incy)
}

/// y := x
unsafe fn copy_slice<T>(input: &[T], output: &mut [T]) where T: Copy {
    debug_assert_eq!(input.len(), output.len());

    let copy = T::copy();
    let mut n = input.len();
    let mut x = input.as_ptr();
    let incx = &1;
    let mut y = output.as_mut_ptr();
    let incy = &1;

    let max = usize::from(i32::max_value()).extract();
    let offset = isize::from(i32::max_value());

    while n >= max {
        copy(&i32::max_value(), x, incx, y, incy);

        x = x.offset(offset);
        y = y.offset(offset);
        n -= max;
    }

    copy(&i32::from(n).extract(), x, incx, y, incy);
}

/// y := alpha * op(A) * x + beta * y
unsafe fn gemv<T>(
    trans: &Transpose,
    alpha: &T,
    a: SubMat<T>,
    beta: &T,
    x: Col<T>,
    y: ColMut<T>,
) where
    T: Gemv,
{
    debug_assert!(match *trans {
        Transpose::No => {
            a.ncols() == x.nrows() && a.nrows() == y.nrows()
        },
        Transpose::Yes => {
            a.nrows() == x.nrows() && a.ncols() == y.nrows()
        },
    } && x.nrows() != 0);

    let x = x.0;
    let y = (y.0).0;

    let gemv = T::gemv();
    let m = &a.nrows;
    let n = &a.ncols;
    let lda = &a.stride;
    let incx = &*x.stride;
    let incy = &*y.stride;

    let a = *a.data;
    let x = *x.data;
    let y = *y.data;

    gemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

/// C := alpha * op(A) * op(B) + beta * C
unsafe fn gemm<T>(
    transa: &Transpose,
    transb: &Transpose,
    alpha: &T,
    a: SubMat<T>,
    b: SubMat<T>,
    beta: &T,
    c: SubMatMut<T>,
) where
    T: Gemm,
{
    let c = c.0;

    debug_assert!(match (*transa, *transb) {
        (Transpose::No, Transpose::No) => {
            a.ncols == b.nrows &&
                a.ncols != 0 &&
                a.nrows == c.nrows &&
                b.ncols == c.ncols
        },
        (Transpose::No, Transpose::Yes) => {
            a.ncols == b.ncols &&
                a.ncols != 0 &&
                a.nrows == c.nrows &&
                b.nrows == c.ncols
        },
        (Transpose::Yes, Transpose::No) => {
            a.nrows == b.nrows &&
                a.nrows != 0 &&
                a.ncols == c.nrows &&
                b.ncols == c.ncols
        },
        (Transpose::Yes, Transpose::Yes) => {
            a.nrows == b.ncols &&
                a.nrows != 0 &&
                a.ncols == c.nrows &&
                b.nrows == c.ncols
        },
    });

    let gemm = T::gemm();
    let (ref m, ref k) = match *transa {
        Transpose::No => (a.nrows, a.ncols),
        Transpose::Yes => (a.ncols, a.nrows),
    };
    let ref n = match *transb {
        Transpose::No => b.ncols,
        Transpose::Yes => b.nrows,
    };
    let lda = &a.stride;
    let ldb = &b.stride;
    let ldc = &c.stride;

    let a = *a.data;
    let b = *b.data;
    let c = *c.data;

    gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/// x := alpha * x
fn scal_slice<A, T>(alpha: &A, x: &mut [T]) where T: Scal<A> {
    unsafe {
        let scal = T::scal();
        let mut n = x.len();
        let mut x = x.as_mut_ptr();
        let ref incx = 1;

        let max = usize::from(i32::max_value()).extract();
        let offset = isize::from(i32::max_value());

        while n >= max {
            scal(&i32::max_value(), alpha, x, incx);

            x = x.offset(offset);
            n -= max;
        }

        scal(&i32::from(n).extract(), alpha, x, incx);
    }
}

/// x := alpha * x
unsafe fn scal_strided<A, T>(alpha: &A, x: &mut Slice<T>) where
    T: Scal<A>,
{
    let scal = T::scal();
    let ref incx = *x.stride;
    let ref n = x.len;
    let x = *x.data;

    scal(n, alpha, x, incx);
}

/// y := alpha * op(A) * x
pub unsafe fn row_mul_submat<T>(
    transa: &Transpose,
    alpha: &T,
    a: SubMat<T>,
    x: Row<T>,
) -> RowVec<T> where
    T: Gemv + Zero,
{
    let x = x.t();
    let ref transa = match *transa {
        Transpose::No => Transpose::Yes,
        Transpose::Yes => Transpose::No,
    };

    submat_mul_col(transa, alpha, a, x).t()
}

/// y := alpha * op(A) * op(B)
pub unsafe fn submat_mul_submat<T>(
    transa: &Transpose,
    transb: &Transpose,
    alpha: &T,
    a: SubMat<T>,
    b: SubMat<T>,
) -> Mat<T> where
    T: Gemm + Zero,
{
    let mut c = Mat::uninitialized(match (*transa, *transb) {
        (Transpose::No, Transpose::No) => (a.nrows, b.ncols),
        (Transpose::No, Transpose::Yes) => (a.nrows, b.nrows),
        (Transpose::Yes, Transpose::No) => (a.ncols, b.ncols),
        (Transpose::Yes, Transpose::Yes) => (a.ncols, b.nrows),
    });

    gemm(transa, transb, alpha, a, b, &T::zero(), c.slice_mut(..));

    c
}

/// y := alpha * op(A) * x
pub unsafe fn submat_mul_col<T>(
    transa: &Transpose,
    alpha: &T,
    a: SubMat<T>,
    x: Col<T>,
) -> ColVec<T> where
    T: Gemv + Zero,
{
    let mut c = ColVec(Tor::uninitialized(match *transa {
        Transpose::No => a.nrows,
        Transpose::Yes => a.ncols,
    }));

    gemv(transa, alpha, a, &T::zero(), x, c.slice_mut(..));

    c
}
