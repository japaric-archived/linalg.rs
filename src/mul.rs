use onezero::{One, Zero};
use std::num::Int;

use blas::{Dot, Gemm, Gemv, ToBlasInt, Transpose, Vector, mod};
use strided;
use traits::{Collection, Matrix};
use {Col, Mat, MutView, Row, Trans, View};

macro_rules! rc1 {
    ($col:ty) => {
        impl<'a, 'b, T> Mul<$col, T> for &'b Row<Box<[T]>> where T: Dot + Zero {
            fn mul(self, rhs: $col) -> T {
                rc(self, rhs)
            }
        }

        impl<'a, 'b, T> Mul<$col, T> for Row<&'b [T]> where T: Dot + Zero {
            fn mul(self, rhs: $col) -> T {
                rc(self, rhs)
            }
        }

        impl<'a, 'b, T> Mul<$col, T> for Row<strided::Slice<'b, T>> where T: Dot + Zero {
            fn mul(self, rhs: $col) -> T {
                rc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<$col, T> for &'b Row<&'c mut [T]> where T: Dot + Zero {
            fn mul(self, rhs: $col) -> T {
                rc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<$col, T> for &'b Row<strided::MutSlice<'c, T>> where
            T: Dot + Zero,
        {
            fn mul(self, rhs: $col) -> T {
                rc(self, rhs)
            }
        }
    }
}

rc1!(&'a Col<Box<[T]>>);
rc1!(Col<&'a [T]>);
rc1!(Col<strided::Slice<'a, T>>);

macro_rules! rc2 {
    ($col:ty) => {
        impl<'a, 'b, 'c, T> Mul<$col, T> for &'c Row<Box<[T]>> where T: Dot + Zero {
            fn mul(self, rhs: $col) -> T {
                rc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<$col, T> for Row<&'c [T]> where T: Dot + Zero {
            fn mul(self, rhs: $col) -> T {
                rc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<$col, T> for Row<strided::Slice<'c, T>> where T: Dot + Zero {
            fn mul(self, rhs: $col) -> T {
                rc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, 'd, T> Mul<$col, T> for &'c Row<&'d mut [T]> where T: Dot + Zero {
            fn mul(self, rhs: $col) -> T {
                rc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, 'd, T> Mul<$col, T> for &'c Row<strided::MutSlice<'d, T>> where
            T: Dot + Zero,
        {
            fn mul(self, rhs: $col) -> T {
                rc(self, rhs)
            }
        }
    }
}

rc2!(&'a Col<&'b mut [T]>);
rc2!(&'a Col<strided::MutSlice<'b, T>>);

macro_rules! mc1 {
    ($col:ty) => {
        impl<'a, 'b, T> Mul<$col, Col<Box<[T]>>> for &'b Mat<T> where T: Gemv + One + Zero {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<$col, Col<Box<[T]>>> for &'b MutView<'c, T> where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, 'b, T> Mul<$col, Col<Box<[T]>>> for &'b Trans<Mat<T>> where T: Gemv + One + Zero {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<$col, Col<Box<[T]>>> for &'b Trans<MutView<'c, T>> where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, 'b, T> Mul<$col, Col<Box<[T]>>> for Trans<View<'b, T>> where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, 'b, T> Mul<$col, Col<Box<[T]>>> for View<'b, T> where T: Gemv + One + Zero {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }
    }
}

mc1!(Col<&'a [T]>);
mc1!(Col<strided::Slice<'a, T>>);
mc1!(&'a Col<Box<[T]>>);

macro_rules! mc2 {
    ($col:ty) => {
        impl<'a, 'b, 'c, T> Mul<$col, Col<Box<[T]>>> for &'c Mat<T> where T: Gemv + One + Zero {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<$col, Col<Box<[T]>>> for &'c Trans<Mat<T>> where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, 'd, T> Mul<$col, Col<Box<[T]>>> for &'c Trans<MutView<'d, T>> where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<$col, Col<Box<[T]>>> for Trans<View<'c, T>> where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<$col, Col<Box<[T]>>> for View<'c, T> where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }

        impl<'a, 'b, 'c, 'd, T> Mul<$col, Col<Box<[T]>>> for &'c MutView<'d, T> where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: $col) -> Col<Box<[T]>> {
                mc(self, rhs)
            }
        }
    }
}

mc2!(&'a Col<&'b mut [T]>);
mc2!(&'a Col<strided::MutSlice<'b, T>>);

macro_rules! mm2 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T> Mul<$rhs, Mat<T>> for $lhs where
            T: Gemm + One + Zero,
        {
            fn mul(self, rhs: $rhs) -> Mat<T> {
                mm(self, rhs)
            }
        }
    }
}

mm2!(&'a Mat<T>, &'b Mat<T>);
mm2!(&'a Trans<Mat<T>>, &'b Trans<Mat<T>>);
mm2!(View<'a, T>, View<'b, T>);
mm2!(Trans<View<'a, T>>, Trans<View<'b, T>>);

macro_rules! mm4 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, 'c, 'd, T> Mul<$rhs, Mat<T>> for $lhs where
            T: Gemm + One + Zero,
        {
            fn mul(self, rhs: $rhs) -> Mat<T> {
                mm(self, rhs)
            }
        }
    }
}

mm4!(&'a MutView<'b, T>, &'c MutView<'d, T>);
mm4!(&'a Trans<MutView<'b, T>>, &'c Trans<MutView<'d, T>>);

macro_rules! cmm2 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, T> Mul<$rhs, Mat<T>> for $lhs where
            T: Gemm + One + Zero,
        {
            fn mul(self, rhs: $rhs) -> Mat<T> {
                mm(self, rhs)
            }
        }

        impl<'a, 'b, T> Mul<$lhs, Mat<T>> for $rhs where
            T: Gemm + One + Zero,
        {
            fn mul(self, rhs: $lhs) -> Mat<T> {
                mm(self, rhs)
            }
        }
    }
}

cmm2!(&'a Mat<T>, &'b Trans<Mat<T>>);
cmm2!(&'a Mat<T>, Trans<View<'b, T>>);
cmm2!(&'a Mat<T>, View<'b, T>);
cmm2!(View<'a, T>, &'b Trans<Mat<T>>);
cmm2!(View<'a, T>, Trans<View<'b, T>>);
cmm2!(&'a Trans<Mat<T>>, Trans<View<'b, T>>);

macro_rules! cmm3 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, 'c, T> Mul<$rhs, Mat<T>> for $lhs where
            T: Gemm + One + Zero,
        {
            fn mul(self, rhs: $rhs) -> Mat<T> {
                mm(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<$lhs, Mat<T>> for $rhs where
            T: Gemm + One + Zero,
        {
            fn mul(self, rhs: $lhs) -> Mat<T> {
                mm(self, rhs)
            }
        }
    }
}

cmm3!(&'a Mat<T>, &'b MutView<'c, T>);
cmm3!(&'a Mat<T>, &'b Trans<MutView<'c, T>>);
cmm3!(View<'a, T>, &'b MutView<'c, T>);
cmm3!(View<'a, T>, &'b Trans<MutView<'c, T>>);
cmm3!(&'a MutView<'b, T>, &'c Trans<Mat<T>>);
cmm3!(&'a MutView<'b, T>, Trans<View<'c, T>>);
cmm3!(&'a Trans<Mat<T>>, &'b Trans<MutView<'c, T>>);
cmm3!(Trans<View<'a, T>>, &'b Trans<MutView<'c, T>>);

macro_rules! cmm4 {
    ($lhs:ty, $rhs:ty) => {
        impl<'a, 'b, 'c, 'd, T> Mul<$rhs, Mat<T>> for $lhs where
            T: Gemm + One + Zero,
        {
            fn mul(self, rhs: $rhs) -> Mat<T> {
                mm(self, rhs)
            }
        }

        impl<'a, 'b, 'c, 'd, T> Mul<$lhs, Mat<T>> for $rhs where
            T: Gemm + One + Zero,
        {
            fn mul(self, rhs: $lhs) -> Mat<T> {
                mm(self, rhs)
            }
        }
    }
}

cmm4!(&'a MutView<'b, T>, &'c Trans<MutView<'d, T>>);

macro_rules! rm2 {
    ($row:ty) => {
        impl<'a, 'b, T> Mul<&'b Mat<T>, Row<Box<[T]>>> for $row where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: &Mat<T>) -> Row<Box<[T]>> {
                rm(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<&'b MutView<'c, T>, Row<Box<[T]>>> for $row where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: &MutView<T>) -> Row<Box<[T]>> {
                rm(self, rhs)
            }
        }

        impl<'a, 'b, T> Mul<&'b Trans<Mat<T>>, Row<Box<[T]>>> for $row where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: &Trans<Mat<T>>) -> Row<Box<[T]>> {
                rm(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<&'b Trans<MutView<'c, T>>, Row<Box<[T]>>> for $row where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: &Trans<MutView<T>>) -> Row<Box<[T]>> {
                rm(self, rhs)
            }
        }

        impl<'a, 'b, T> Mul<Trans<View<'b, T>>, Row<Box<[T]>>> for $row where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: Trans<View<T>>) -> Row<Box<[T]>> {
                rm(self, rhs)
            }
        }

        impl<'a, 'b, T> Mul<View<'b, T>, Row<Box<[T]>>> for $row where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: View<T>) -> Row<Box<[T]>> {
                rm(self, rhs)
            }
        }
    }
}

rm2!(&'a Row<Box<[T]>>);
rm2!(Row<&'a [T]>);
rm2!(Row<strided::Slice<'a, T>>);

macro_rules! rm3 {
    ($row:ty) => {
        impl<'a, 'b, 'c, T> Mul<&'c Mat<T>, Row<Box<[T]>>> for $row where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: &Mat<T>) -> Row<Box<[T]>> {
                rm(self, rhs)
            }
        }

        impl<'a, 'b, 'c, 'd, T> Mul<&'c MutView<'d, T>, Row<Box<[T]>>> for $row where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: &MutView<T>) -> Row<Box<[T]>> {
                rm(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<&'c Trans<Mat<T>>, Row<Box<[T]>>> for $row where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: &Trans<Mat<T>>) -> Row<Box<[T]>> {
                rm(self, rhs)
            }
        }

        impl<'a, 'b, 'c, T> Mul<View<'c, T>, Row<Box<[T]>>> for $row where
            T: Gemv + One + Zero,
        {
            fn mul(self, rhs: View<T>) -> Row<Box<[T]>> {
                rm(self, rhs)
            }
        }
    }
}

rm2!(&'a Row<&'b mut [T]>);
rm2!(&'a Row<strided::MutSlice<'b, T>>);

fn mc<T, M, V>(lhs: M, rhs: V) -> Col<Box<[T]>> where
    T: Gemv + One + Zero,
    M: blas::Matrix<T>,
    V: Vector<T>,
{
    assert_eq!(lhs.ncols(), Collection::len(&rhs));
    assert!(lhs.ncols() != 0);

    if lhs.nrows() == 0 { return Col::new(box []) }

    let gemv = Gemv::gemv(None::<T>);
    let trans = lhs.trans();
    let (m, n) = match trans {
        Transpose::No => (lhs.nrows().to_blasint(), lhs.ncols().to_blasint()),
        Transpose::Yes => (lhs.ncols().to_blasint(), lhs.nrows().to_blasint()),
    };
    let alpha = One::one();
    let a = lhs.as_ptr();
    let lda = lhs.stride().unwrap_or(m);
    let x = rhs.as_ptr();
    let incx = rhs.stride();
    let beta = Zero::zero();
    let mut data = Vec::with_capacity(lhs.nrows());
    let y = data.as_mut_ptr();
    let incy = One::one();

    unsafe {
        gemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
        data.set_len(lhs.nrows());
    }

    Col::new(data.into_boxed_slice())
}

fn mm<T, L, R>(lhs: L, rhs: R) -> Mat<T> where
    T: Gemm + One + Zero,
    L: blas::Matrix<T>,
    R: blas::Matrix<T>,
{
    assert_eq!(lhs.ncols(), rhs.nrows());
    assert!(lhs.ncols() != 0);

    if lhs.nrows() == 0 || rhs.ncols() == 0 {
        return unsafe { Mat::from_parts(box [], (lhs.nrows(), rhs.ncols())) }
    }

    let length = lhs.nrows().checked_mul(rhs.ncols()).unwrap();

    let gemm = Gemm::gemm(None::<T>);
    let transa = lhs.trans();
    let transb = rhs.trans();
    let m = lhs.nrows().to_blasint();
    let k = lhs.ncols().to_blasint();
    let n = rhs.ncols().to_blasint();
    let alpha = One::one();
    let a = lhs.as_ptr();
    let lda = lhs.stride().unwrap_or_else(|| match transa {
        Transpose::No => m,
        Transpose::Yes => k,
    });
    let b = rhs.as_ptr();
    let ldb = rhs.stride().unwrap_or_else(|| match transb {
        Transpose::No => k,
        Transpose::Yes => n,
    });
    let beta = Zero::zero();
    let mut data = Vec::with_capacity(length);
    let c = data.as_mut_ptr();
    let ldc = m;

    unsafe {
        gemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
        data.set_len(length);
        Mat::from_parts(data.into_boxed_slice(), (lhs.nrows(), rhs.ncols()))
    }
}

fn rc<T, R, C>(lhs: R, rhs: C) -> T where
    T: Dot + Zero,
    R: Vector<T>,
    C: Vector<T>,
{
    let lhs_ncols = Collection::len(&lhs);
    let rhs_nrows = Collection::len(&rhs);

    assert_eq!(lhs_ncols, rhs_nrows);

    let n = lhs_ncols;

    if n == 0 { return Zero::zero() }

    let dot = Dot::dot(None::<T>);
    let n = n.to_blasint();
    let x = lhs.as_ptr();
    let incx = lhs.stride();
    let y = rhs.as_ptr();
    let incy = rhs.stride();

    unsafe {
        dot(&n, x, &incx, y, &incy)
    }
}

fn rm<T, M, V>(lhs: V, rhs: M) -> Row<Box<[T]>> where
    T: Gemv + One + Zero,
    M: blas::Matrix<T>,
    V: Vector<T>,
{
    let lhs_ncols = Collection::len(&lhs);
    assert_eq!(lhs_ncols, rhs.nrows());
    assert!(lhs_ncols != 0);

    if rhs.ncols() == 0 { return Row::new(box []) }

    let gemv = Gemv::gemv(None::<T>);
    let trans = !rhs.trans();
    let (m, n) = match trans {
        Transpose::No => (rhs.ncols().to_blasint(), rhs.nrows().to_blasint()),
        Transpose::Yes => (rhs.nrows().to_blasint(), rhs.ncols().to_blasint()),
    };
    let alpha = One::one();
    let a = rhs.as_ptr();
    let lda = rhs.stride().unwrap_or(m);
    let x = lhs.as_ptr();
    let incx = lhs.stride();
    let beta = Zero::zero();
    let mut data = Vec::with_capacity(rhs.ncols());
    let y = data.as_mut_ptr();
    let incy = One::one();

    unsafe {
        gemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
        data.set_len(rhs.ncols());
    }

    Row::new(data.into_boxed_slice())
}
