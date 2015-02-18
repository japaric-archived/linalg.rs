use cast::CastTo;
use onezero::{One, Zero};
use std::num::Int;
use std::ops::Mul;

use blas::{blasint, Transpose};
use blas::dot::Dot;
use blas::gemm::Gemm;
use blas::gemv::Gemv;
use traits::Matrix;
use {Col, ColVec, Mat, MutCol, MutRow, MutView, Row, RowVec, Trans, View};

// mat * col (gemv)
fn mc<T>(
    trans: Transpose,
    lhs: ::raw::View<T>,
    rhs: ::raw::strided::Slice<T>,
    len: usize,
) -> Box<[T]> where
    T: Gemv + One + Zero,
{
    if len == 0 {
        Box::new([])
    } else {
        let mut data = Vec::with_capacity(len);

        let gemv = <T as Gemv>::gemv();
        let m = lhs.nrows.to::<blasint>().unwrap();
        let n = lhs.ncols.to::<blasint>().unwrap();
        let alpha = One::one();
        let a = lhs.data;
        let lda = lhs.ld.to::<blasint>().unwrap();
        let x = rhs.data;
        let incx = rhs.stride.to::<blasint>().unwrap();
        let beta = Zero::zero();
        let y = data.as_mut_ptr();
        let incy = One::one();

        unsafe {
            gemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
            data.set_len(len)
        }

        data.into_boxed_slice()
    }
}

impl<'a, 'b, T> Mul<Col<'a, T>> for &'b Mat<T> where T: Gemv + One + Zero {
    type Output = ColVec<T>;

    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        self.as_view() * rhs
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>> for &'b Mat<T> where T: Gemv + One + Zero {
    type Output = ColVec<T>;

    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        self.as_view() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>> for &'c Mat<T> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        self.as_view() * *rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<Col<'a, T>> for &'b MutView<'c, T> where T: Gemv + One + Zero {
    type Output = ColVec<T>;

    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        *self.as_view() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<&'a ColVec<T>> for &'b MutView<'c, T> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        *self.as_view() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutCol<'b, T>> for &'c MutView<'d, T> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        *self.as_view() * *rhs.as_col()
    }
}

impl<'a, 'b, T> Mul<Col<'a, T>> for &'b Trans<Mat<T>> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        self.as_trans_view_() * rhs
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>> for &'b Trans<Mat<T>> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        self.as_trans_view_() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>> for &'c Trans<Mat<T>> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        self.as_trans_view_() * *rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<Col<'a, T>> for &'b Trans<MutView<'c, T>> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        *self.as_trans_view() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<&'a ColVec<T>> for &'b Trans<MutView<'c, T>> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        *self.as_trans_view() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutCol<'b, T>> for &'c Trans<MutView<'d, T>> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        *self.as_trans_view() * *rhs.as_col()
    }
}

impl<'a, 'b, T> Mul<Col<'a, T>> for Trans<View<'b, T>> where T: Gemv + One + Zero {
    type Output = ColVec<T>;

    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.len());
        assert!(rhs.len() != 0);

        ColVec::new(mc(Transpose::Yes, (self.0).0, (rhs.0).0, lhs.nrows()))
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>> for Trans<View<'b, T>> where T: Gemv + One + Zero {
    type Output = ColVec<T>;

    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        self * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>> for Trans<View<'c, T>> where
    T: Gemv + One + Zero,
{
    type Output = ColVec<T>;

    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        self * *rhs.as_col()
    }
}

impl<'a, 'b, T> Mul<Col<'a, T>> for View<'b, T> where T: Gemv + One + Zero {
    type Output = ColVec<T>;

    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.len());
        assert!(rhs.len() != 0);

        ColVec::new(mc(Transpose::No, lhs.0, (rhs.0).0, lhs.nrows()))
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>> for View<'b, T> where T: Gemv + One + Zero {
    type Output = ColVec<T>;

    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        self * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>> for View<'c, T> where T: Gemv + One + Zero {
    type Output = ColVec<T>;

    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        self * *rhs.as_col()
    }
}

// mat * mat (gemm)
fn mm<T>(
    transa: Transpose,
    lhs: ::raw::View<T>,
    transb: Transpose,
    rhs: ::raw::View<T>,
    size: (usize, usize),
) -> Mat<T> where
    T: Gemm + One + Zero,
{
    let data: Box<[_]> = if size.0 == 0 || size.1 == 0 {
        Box::new([])
    } else {
        let len = size.0.checked_mul(size.1).unwrap();

        let gemm = <T as Gemm>::gemm();
        let (m, k) = match transa {
            Transpose::No => {
                (lhs.nrows.to::<blasint>().unwrap(), lhs.ncols.to::<blasint>().unwrap())
            },
            Transpose::Yes => {
                (lhs.ncols.to::<blasint>().unwrap(), lhs.nrows.to::<blasint>().unwrap())
            },
        };
        let n = match transb {
            Transpose::No => rhs.ncols.to::<blasint>().unwrap(),
            Transpose::Yes => rhs.nrows.to::<blasint>().unwrap(),
        };
        let alpha = One::one();
        let a = lhs.data;
        let lda = lhs.ld.to::<blasint>().unwrap();
        let b = rhs.data;
        let ldb = rhs.ld.to::<blasint>().unwrap();
        let beta = Zero::zero();
        let mut data = Vec::with_capacity(len);
        let c = data.as_mut_ptr();
        let ldc = m;

        unsafe {
            gemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
            data.set_len(len);
        }

        data.into_boxed_slice()
    };

    unsafe {
        Mat::from_parts(data, size)
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>> for &'b Mat<T> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        self.as_view() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>> for &'c Mat<T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        self.as_view() * *rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>> for &'b Mat<T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        self.as_view() * rhs.as_trans_view_()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>> for &'c Mat<T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        self.as_view() * *rhs.as_trans_view()
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>> for &'b Mat<T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        self.as_view() * rhs
    }
}

impl<'a, 'b, T> Mul<View<'a, T>> for &'b Mat<T> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: View<T>) -> Mat<T> {
        self.as_view() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Mat<T>> for &'b MutView<'c, T> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        *self.as_view() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutView<'b, T>> for &'c MutView<'d, T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        *self.as_view() * *rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<Mat<T>>> for &'b MutView<'c, T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        *self.as_view() * rhs.as_trans_view_()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a Trans<MutView<'b, T>>> for &'c MutView<'d, T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        *self.as_view() * *rhs.as_trans_view()
    }
}

impl<'a, 'b, 'c, T> Mul<Trans<View<'a, T>>> for &'b MutView<'c, T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        *self.as_view() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<View<'a, T>> for &'b MutView<'c, T> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: View<T>) -> Mat<T> {
        *self.as_view() * rhs
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>> for &'b Trans<Mat<T>> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        self.as_trans_view_() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>> for &'c Trans<Mat<T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        self.as_trans_view_() * *rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>> for &'b Trans<Mat<T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        self.as_trans_view_() * rhs.as_trans_view_()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>> for &'c Trans<Mat<T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        self.as_trans_view_() * *rhs.as_trans_view()
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>> for &'b Trans<Mat<T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        self.as_trans_view_() * rhs
    }
}

impl<'a, 'b, T> Mul<View<'a, T>> for &'b Trans<Mat<T>> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: View<T>) -> Mat<T> {
        self.as_trans_view_() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Mat<T>> for &'b Trans<MutView<'c, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        *self.as_trans_view() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutView<'b, T>> for &'c Trans<MutView<'d, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        *self.as_trans_view() * *rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<Mat<T>>> for &'b Trans<MutView<'c, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        *self.as_trans_view() * rhs.as_trans_view_()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a Trans<MutView<'b, T>>> for &'c Trans<MutView<'d, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        *self.as_trans_view() * *rhs.as_trans_view()
    }
}

impl<'a, 'b, 'c, T> Mul<Trans<View<'a, T>>> for &'b Trans<MutView<'c, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        *self.as_trans_view() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<View<'a, T>> for &'b Trans<MutView<'c, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: View<T>) -> Mat<T> {
        *self.as_trans_view() * rhs
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>> for Trans<View<'b, T>> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        self * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>> for Trans<View<'c, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        self * *rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>> for Trans<View<'b, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        self * rhs.as_trans_view_()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>> for Trans<View<'c, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        self * *rhs.as_trans_view()
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>> for Trans<View<'b, T>> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.nrows());
        assert!(lhs.ncols() != 0);

        mm(Transpose::Yes, (lhs.0).0, Transpose::Yes, (rhs.0).0, (lhs.nrows(), rhs.ncols()))
    }
}

impl<'a, 'b, T> Mul<View<'a, T>> for Trans<View<'b, T>> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: View<T>) -> Mat<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.nrows());
        assert!(lhs.ncols() != 0);

        mm(Transpose::Yes, (lhs.0).0, Transpose::No, rhs.0, (lhs.nrows(), rhs.ncols()))
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>> for View<'b, T> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        self * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>> for View<'c, T> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        self * *rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>> for View<'b, T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        self * rhs.as_trans_view_()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>> for View<'c, T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        self * *rhs.as_trans_view()
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>> for View<'b, T> where
    T: Gemm + One + Zero,
{
    type Output = Mat<T>;

    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.nrows());
        assert!(lhs.ncols() != 0);

        mm(Transpose::No, lhs.0, Transpose::Yes, (rhs.0).0, (lhs.nrows(), rhs.ncols()))
    }
}

impl<'a, 'b, T> Mul<View<'a, T>> for View<'b, T> where T: Gemm + One + Zero {
    type Output = Mat<T>;

    fn mul(self, rhs: View<T>) -> Mat<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.nrows());
        assert!(lhs.ncols() != 0);

        mm(Transpose::No, lhs.0, Transpose::No, rhs.0, (lhs.nrows(), rhs.ncols()))
    }
}

// row * col (dot product)
impl<'a, 'b, T> Mul<Col<'a, T>> for &'b RowVec<T> where T: Dot + Zero {
    type Output = T;

    fn mul(self, rhs: Col<T>) -> T {
        self.as_row() * rhs
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>> for &'b RowVec<T> where T: Dot + Zero {
    type Output = T;

    fn mul(self, rhs: &ColVec<T>) -> T {
        self.as_row() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>> for &'c RowVec<T> where T: Dot + Zero {
    type Output = T;

    fn mul(self, rhs: &MutCol<T>) -> T {
        self.as_row() * *rhs.as_col()
    }
}

impl<'a, 'b, T> Mul<Col<'a, T>> for Row<'b, T> where T: Dot + Zero {
    type Output = T;

    fn mul(self, rhs: Col<T>) -> T {
        assert_eq!(self.len(), rhs.len());

        let n = self.len();

        if n == 0 { return Zero::zero() }

        let (lhs, rhs) = ((self.0).0, (rhs.0).0);

        let dot = <T as Dot>::dot();
        let x = lhs.data;
        let incx = lhs.stride.to::<blasint>().unwrap();
        let y = rhs.data;
        let incy = rhs.stride.to::<blasint>().unwrap();

        unsafe {
            dot(&n.to::<blasint>().unwrap(), x, &incx, y, &incy)
        }
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>> for Row<'b, T> where T: Dot + Zero {
    type Output = T;

    fn mul(self, rhs: &ColVec<T>) -> T {
        self * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>> for Row<'c, T> where T: Dot + Zero {
    type Output = T;

    fn mul(self, rhs: &MutCol<T>) -> T {
        self * *rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<Col<'a, T>> for &'b MutRow<'c, T> where T: Dot + Zero {
    type Output = T;

    fn mul(self, rhs: Col<T>) -> T {
        *self.as_row() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<&'a ColVec<T>> for &'b MutRow<'c, T> where T: Dot + Zero {
    type Output = T;

    fn mul(self, rhs: &ColVec<T>) -> T {
        *self.as_row() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutCol<'b, T>> for &'c MutRow<'d, T> where T: Dot + Zero {
    type Output = T;

    fn mul(self, rhs: &MutCol<T>) -> T {
        *self.as_row() * *rhs.as_col()
    }
}

// row * mat (gemv)
impl<'a, 'b, 'c, T> Mul<&'a Mat<T>> for &'b MutRow<'c, T> where T: Gemv + One + Zero {
    type Output = RowVec<T>;

    fn mul(self, rhs: &Mat<T>) -> RowVec<T> {
        *self.as_row() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutView<'b, T>> for &'c MutRow<'d, T> where
    T: Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn mul(self, rhs: &MutView<T>) -> RowVec<T> {
        *self.as_row() * *rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<Mat<T>>> for &'b MutRow<'c, T> where
    T: Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn mul(self, rhs: &Trans<Mat<T>>) -> RowVec<T> {
        *self.as_row() * rhs.as_trans_view_()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a Trans<MutView<'b, T>>> for &'c MutRow<'d, T> where
    T: Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn mul(self, rhs: &Trans<MutView<T>>) -> RowVec<T> {
        *self.as_row() * *rhs.as_trans_view()
    }
}

impl<'a, 'b, 'c, T> Mul<Trans<View<'a, T>>> for &'b MutRow<'c, T> where
    T: Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn mul(self, rhs: Trans<View<T>>) -> RowVec<T> {
        *self.as_row() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<View<'a, T>> for &'b MutRow<'c, T> where T: Gemv + One + Zero {
    type Output = RowVec<T>;

    fn mul(self, rhs: View<T>) -> RowVec<T> {
        *self.as_row() * rhs
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>> for Row<'b, T> where T: Gemv + One + Zero {
    type Output = RowVec<T>;

    fn mul(self, rhs: &Mat<T>) -> RowVec<T> {
        self * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>> for Row<'c, T> where T: Gemv + One + Zero {
    type Output = RowVec<T>;

    fn mul(self, rhs: &MutView<T>) -> RowVec<T> {
        self * *rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>> for Row<'b, T> where
    T: Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn mul(self, rhs: &Trans<Mat<T>>) -> RowVec<T> {
        self * rhs.as_trans_view_()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>> for Row<'c, T> where
    T: Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn mul(self, rhs: &Trans<MutView<T>>) -> RowVec<T> {
        self * *rhs.as_trans_view()
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>> for Row<'b, T> where T: Gemv + One + Zero {
    type Output = RowVec<T>;

    fn mul(self, rhs: Trans<View<T>>) -> RowVec<T> {
        let lhs = self;

        assert_eq!(lhs.len(), rhs.nrows());
        assert!(lhs.len() != 0);

        // NB (mat * row)^T == col * mat^T
        RowVec::new(mc(Transpose::No, (rhs.0).0, (lhs.0).0, rhs.ncols()))
    }
}

impl<'a, 'b, T> Mul<View<'a, T>> for Row<'b, T> where T: Gemv + One + Zero {
    type Output = RowVec<T>;

    fn mul(self, rhs: View<T>) -> RowVec<T> {
        let lhs = self;

        assert_eq!(lhs.len(), rhs.nrows());
        assert!(lhs.len() != 0);

        // NB (mat * row)^T == col * mat^T
        RowVec::new(mc(Transpose::Yes, rhs.0, (lhs.0).0, rhs.ncols()))
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>> for &'b RowVec<T> where T: Gemv + One + Zero {
    type Output = RowVec<T>;

    fn mul(self, rhs: &Mat<T>) -> RowVec<T> {
        self.as_row() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>> for &'c RowVec<T> where
    T: Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn mul(self, rhs: &MutView<T>) -> RowVec<T> {
        self.as_row() * *rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>> for &'b RowVec<T> where
    T: Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn mul(self, rhs: &Trans<Mat<T>>) -> RowVec<T> {
        self.as_row() * rhs.as_trans_view_()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>> for &'c RowVec<T> where
    T: Gemv + One + Zero,
{
    type Output = RowVec<T>;

    fn mul(self, rhs: &Trans<MutView<T>>) -> RowVec<T> {
        self.as_row() * *rhs.as_trans_view()
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>> for &'b RowVec<T> where T: Gemv + One + Zero {
    type Output = RowVec<T>;

    fn mul(self, rhs: Trans<View<T>>) -> RowVec<T> {
        self.as_row() * rhs
    }
}

impl<'a, 'b, T> Mul<View<'a, T>> for &'b RowVec<T> where T: Gemv + One + Zero {
    type Output = RowVec<T>;

    fn mul(self, rhs: View<T>) -> RowVec<T> {
        self.as_row() * rhs
    }
}
