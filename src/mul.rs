use onezero::{One, Zero};
use std::num::Int;
use std::ops::Mul;

use blas::Transpose;
use blas::dot::Dot;
use blas::gemm::Gemm;
use blas::gemv::Gemv;
use traits::Matrix;
use {Col, ColVec, Mat, MutCol, MutRow, MutView, Row, RowVec, ToBlasint, Trans, View};

// mat * col (gemv)
fn mc<T>(
    trans: Transpose,
    lhs: ::raw::View<T>,
    rhs: ::raw::strided::Slice<T>,
    len: uint,
) -> Box<[T]> where
    T: Gemv + One + Zero,
{
    if len == 0 {
        box []
    } else {
        let mut data = Vec::with_capacity(len);

        let gemv = Gemv::gemv(None::<T>);
        let m = lhs.nrows.to_blasint();
        let n = lhs.ncols.to_blasint();
        let alpha = One::one();
        let a = lhs.data;
        let lda = lhs.ld.to_blasint();
        let x = rhs.data;
        let incx = rhs.stride.to_blasint();
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

impl<'a, 'b, T> Mul<Col<'a, T>, ColVec<T>> for &'b Mat<T> where T: Gemv + One + Zero {
    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        self.as_view() * rhs
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>, ColVec<T>> for &'b Mat<T> where T: Gemv + One + Zero {
    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        self.as_view() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>, ColVec<T>> for &'c Mat<T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        self.as_view() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<Col<'a, T>, ColVec<T>> for &'b MutView<'c, T> where T: Gemv + One + Zero {
    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        self.as_view() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<&'a ColVec<T>, ColVec<T>> for &'b MutView<'c, T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        self.as_view() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutCol<'b, T>, ColVec<T>> for &'c MutView<'d, T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        self.as_view() * rhs.as_col()
    }
}

impl<'a, 'b, T> Mul<Col<'a, T>, ColVec<T>> for &'b Trans<Mat<T>> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        Trans(self.0.as_view()) * rhs
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>, ColVec<T>> for &'b Trans<Mat<T>> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        Trans(self.0.as_view()) * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>, ColVec<T>> for &'c Trans<Mat<T>> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        Trans(self.0.as_view()) * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<Col<'a, T>, ColVec<T>> for &'b Trans<MutView<'c, T>> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        Trans(self.0.as_view())* rhs
    }
}

impl<'a, 'b, 'c, T> Mul<&'a ColVec<T>, ColVec<T>> for &'b Trans<MutView<'c, T>> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        Trans(self.0.as_view())* rhs.as_col()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutCol<'b, T>, ColVec<T>> for &'c Trans<MutView<'d, T>> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        Trans(self.0.as_view())* rhs.as_col()
    }
}

impl<'a, 'b, T> Mul<Col<'a, T>, ColVec<T>> for Trans<View<'b, T>> where T: Gemv + One + Zero {
    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.len());
        assert!(rhs.len() != 0);

        ColVec::new(mc(Transpose::Yes, (self.0).0, (rhs.0).0, lhs.nrows()))
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>, ColVec<T>> for Trans<View<'b, T>> where T: Gemv + One + Zero {
    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        self * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>, ColVec<T>> for Trans<View<'c, T>> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        self * rhs.as_col()
    }
}

impl<'a, 'b, T> Mul<Col<'a, T>, ColVec<T>> for View<'b, T> where T: Gemv + One + Zero {
    fn mul(self, rhs: Col<T>) -> ColVec<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.len());
        assert!(rhs.len() != 0);

        ColVec::new(mc(Transpose::No, lhs.0, (rhs.0).0, lhs.nrows()))
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>, ColVec<T>> for View<'b, T> where T: Gemv + One + Zero {
    fn mul(self, rhs: &ColVec<T>) -> ColVec<T> {
        self * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>, ColVec<T>> for View<'c, T> where T: Gemv + One + Zero {
    fn mul(self, rhs: &MutCol<T>) -> ColVec<T> {
        self * rhs.as_col()
    }
}

// mat * mat (gemm)
fn mm<T>(
    transa: Transpose,
    lhs: ::raw::View<T>,
    transb: Transpose,
    rhs: ::raw::View<T>,
    size: (uint, uint),
) -> Mat<T> where
    T: Gemm + One + Zero,
{
    let data: Box<[_]> = if size.0 == 0 || size.1 == 0 {
        box []
    } else {
        let len = size.0.checked_mul(size.1).unwrap();

        let gemm = Gemm::gemm(None::<T>);
        let (m, k) = match transa {
            Transpose::No => (lhs.nrows.to_blasint(), lhs.ncols.to_blasint()),
            Transpose::Yes => (lhs.ncols.to_blasint(), lhs.nrows.to_blasint()),
        };
        let n = match transb {
            Transpose::No => rhs.ncols.to_blasint(),
            Transpose::Yes => rhs.nrows.to_blasint(),
        };
        let alpha = One::one();
        let a = lhs.data;
        let lda = lhs.ld.to_blasint();
        let b = rhs.data;
        let ldb = rhs.ld.to_blasint();
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

impl<'a, 'b, T> Mul<&'a Mat<T>, Mat<T>> for &'b Mat<T> where T: Gemm + One + Zero {
    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        self.as_view() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>, Mat<T>> for &'c Mat<T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        self.as_view() * rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>, Mat<T>> for &'b Mat<T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        self.as_view() * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>, Mat<T>> for &'c Mat<T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        self.as_view() * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>, Mat<T>> for &'b Mat<T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        self.as_view() * rhs
    }
}

impl<'a, 'b, T> Mul<View<'a, T>, Mat<T>> for &'b Mat<T> where T: Gemm + One + Zero {
    fn mul(self, rhs: View<T>) -> Mat<T> {
        self.as_view() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Mat<T>, Mat<T>> for &'b MutView<'c, T> where T: Gemm + One + Zero {
    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        self.as_view() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutView<'b, T>, Mat<T>> for &'c MutView<'d, T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        self.as_view() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<Mat<T>>, Mat<T>> for &'b MutView<'c, T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        self.as_view() * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a Trans<MutView<'b, T>>, Mat<T>> for &'c MutView<'d, T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        self.as_view() * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> Mul<Trans<View<'a, T>>, Mat<T>> for &'b MutView<'c, T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        self.as_view() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<View<'a, T>, Mat<T>> for &'b MutView<'c, T> where T: Gemm + One + Zero {
    fn mul(self, rhs: View<T>) -> Mat<T> {
        self.as_view() * rhs
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>, Mat<T>> for &'b Trans<Mat<T>> where T: Gemm + One + Zero {
    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        Trans(self.0.as_view()) * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>, Mat<T>> for &'c Trans<Mat<T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        Trans(self.0.as_view()) * rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>, Mat<T>> for &'b Trans<Mat<T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        Trans(self.0.as_view()) * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>, Mat<T>> for &'c Trans<Mat<T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        Trans(self.0.as_view()) * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>, Mat<T>> for &'b Trans<Mat<T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        Trans(self.0.as_view()) * rhs
    }
}

impl<'a, 'b, T> Mul<View<'a, T>, Mat<T>> for &'b Trans<Mat<T>> where T: Gemm + One + Zero {
    fn mul(self, rhs: View<T>) -> Mat<T> {
        Trans(self.0.as_view()) * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Mat<T>, Mat<T>> for &'b Trans<MutView<'c, T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        Trans(self.0.as_view()) * rhs.as_view()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutView<'b, T>, Mat<T>> for &'c Trans<MutView<'d, T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        Trans(self.0.as_view()) * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<Mat<T>>, Mat<T>> for &'b Trans<MutView<'c, T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        Trans(self.0.as_view()) * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a Trans<MutView<'b, T>>, Mat<T>> for &'c Trans<MutView<'d, T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        Trans(self.0.as_view()) * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> Mul<Trans<View<'a, T>>, Mat<T>> for &'b Trans<MutView<'c, T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        Trans(self.0.as_view()) * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<View<'a, T>, Mat<T>> for &'b Trans<MutView<'c, T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: View<T>) -> Mat<T> {
        Trans(self.0.as_view()) * rhs
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>, Mat<T>> for Trans<View<'b, T>> where T: Gemm + One + Zero {
    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        self * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>, Mat<T>> for Trans<View<'c, T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        self * rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>, Mat<T>> for Trans<View<'b, T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        self * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>, Mat<T>> for Trans<View<'c, T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        self * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>, Mat<T>> for Trans<View<'b, T>> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.nrows());
        assert!(lhs.ncols() != 0);

        mm(Transpose::Yes, (lhs.0).0, Transpose::Yes, (rhs.0).0, (lhs.nrows(), rhs.ncols()))
    }
}

impl<'a, 'b, T> Mul<View<'a, T>, Mat<T>> for Trans<View<'b, T>> where T: Gemm + One + Zero {
    fn mul(self, rhs: View<T>) -> Mat<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.nrows());
        assert!(lhs.ncols() != 0);

        mm(Transpose::Yes, (lhs.0).0, Transpose::No, rhs.0, (lhs.nrows(), rhs.ncols()))
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>, Mat<T>> for View<'b, T> where T: Gemm + One + Zero {
    fn mul(self, rhs: &Mat<T>) -> Mat<T> {
        self * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>, Mat<T>> for View<'c, T> where T: Gemm + One + Zero {
    fn mul(self, rhs: &MutView<T>) -> Mat<T> {
        self * rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>, Mat<T>> for View<'b, T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<Mat<T>>) -> Mat<T> {
        self * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>, Mat<T>> for View<'c, T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: &Trans<MutView<T>>) -> Mat<T> {
        self * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>, Mat<T>> for View<'b, T> where
    T: Gemm + One + Zero,
{
    fn mul(self, rhs: Trans<View<T>>) -> Mat<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.nrows());
        assert!(lhs.ncols() != 0);

        mm(Transpose::No, lhs.0, Transpose::Yes, (rhs.0).0, (lhs.nrows(), rhs.ncols()))
    }
}

impl<'a, 'b, T> Mul<View<'a, T>, Mat<T>> for View<'b, T> where T: Gemm + One + Zero {
    fn mul(self, rhs: View<T>) -> Mat<T> {
        let lhs = self;

        assert_eq!(lhs.ncols(), rhs.nrows());
        assert!(lhs.ncols() != 0);

        mm(Transpose::No, lhs.0, Transpose::No, rhs.0, (lhs.nrows(), rhs.ncols()))
    }
}

// row * col (dot product)
impl<'a, 'b, T> Mul<Col<'a, T>, T> for &'b RowVec<T> where T: Dot + Zero {
    fn mul(self, rhs: Col<T>) -> T {
        self.as_row() * rhs
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>, T> for &'b RowVec<T> where T: Dot + Zero {
    fn mul(self, rhs: &ColVec<T>) -> T {
        self.as_row() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>, T> for &'c RowVec<T> where T: Dot + Zero {
    fn mul(self, rhs: &MutCol<T>) -> T {
        self.as_row() * rhs.as_col()
    }
}

impl<'a, 'b, T> Mul<Col<'a, T>, T> for Row<'b, T> where T: Dot + Zero {
    fn mul(self, rhs: Col<T>) -> T {
        assert_eq!(self.len(), rhs.len());

        let n = self.len();

        if n == 0 { return Zero::zero() }

        let (lhs, rhs) = ((self.0).0, (rhs.0).0);

        let dot = Dot::dot(None::<T>);
        let x = lhs.data;
        let incx = lhs.stride.to_blasint();
        let y = rhs.data;
        let incy = rhs.stride.to_blasint();

        unsafe {
            dot(&n.to_blasint(), x, &incx, y, &incy)
        }
    }
}

impl<'a, 'b, T> Mul<&'a ColVec<T>, T> for Row<'b, T> where T: Dot + Zero {
    fn mul(self, rhs: &ColVec<T>) -> T {
        self * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutCol<'b, T>, T> for Row<'c, T> where T: Dot + Zero {
    fn mul(self, rhs: &MutCol<T>) -> T {
        self * rhs.as_col()
    }
}

impl<'a, 'b, 'c, T> Mul<Col<'a, T>, T> for &'b MutRow<'c, T> where T: Dot + Zero {
    fn mul(self, rhs: Col<T>) -> T {
        self.as_row() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<&'a ColVec<T>, T> for &'b MutRow<'c, T> where T: Dot + Zero {
    fn mul(self, rhs: &ColVec<T>) -> T {
        self.as_row() * rhs.as_col()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutCol<'b, T>, T> for &'c MutRow<'d, T> where T: Dot + Zero {
    fn mul(self, rhs: &MutCol<T>) -> T {
        self.as_row() * rhs.as_col()
    }
}

// row * mat (gemv)
impl<'a, 'b, 'c, T> Mul<&'a Mat<T>, RowVec<T>> for &'b MutRow<'c, T> where T: Gemv + One + Zero {
    fn mul(self, rhs: &Mat<T>) -> RowVec<T> {
        self.as_row() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a MutView<'b, T>, RowVec<T>> for &'c MutRow<'d, T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &MutView<T>) -> RowVec<T> {
        self.as_row() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<Mat<T>>, RowVec<T>> for &'b MutRow<'c, T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &Trans<Mat<T>>) -> RowVec<T> {
        self.as_row() * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, 'd, T> Mul<&'a Trans<MutView<'b, T>>, RowVec<T>> for &'c MutRow<'d, T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &Trans<MutView<T>>) -> RowVec<T> {
        self.as_row() * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> Mul<Trans<View<'a, T>>, RowVec<T>> for &'b MutRow<'c, T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: Trans<View<T>>) -> RowVec<T> {
        self.as_row() * rhs
    }
}

impl<'a, 'b, 'c, T> Mul<View<'a, T>, RowVec<T>> for &'b MutRow<'c, T> where T: Gemv + One + Zero {
    fn mul(self, rhs: View<T>) -> RowVec<T> {
        self.as_row() * rhs
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>, RowVec<T>> for Row<'b, T> where T: Gemv + One + Zero {
    fn mul(self, rhs: &Mat<T>) -> RowVec<T> {
        self * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>, RowVec<T>> for Row<'c, T> where T: Gemv + One + Zero {
    fn mul(self, rhs: &MutView<T>) -> RowVec<T> {
        self * rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>, RowVec<T>> for Row<'b, T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &Trans<Mat<T>>) -> RowVec<T> {
        self * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>, RowVec<T>> for Row<'c, T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &Trans<MutView<T>>) -> RowVec<T> {
        self * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>, RowVec<T>> for Row<'b, T> where T: Gemv + One + Zero {
    fn mul(self, rhs: Trans<View<T>>) -> RowVec<T> {
        let lhs = self;

        assert_eq!(lhs.len(), rhs.nrows());
        assert!(lhs.len() != 0);

        // NB (mat * row)^T == col * mat^T
        RowVec::new(mc(Transpose::No, (rhs.0).0, (lhs.0).0, rhs.ncols()))
    }
}

impl<'a, 'b, T> Mul<View<'a, T>, RowVec<T>> for Row<'b, T> where T: Gemv + One + Zero {
    fn mul(self, rhs: View<T>) -> RowVec<T> {
        let lhs = self;

        assert_eq!(lhs.len(), rhs.nrows());
        assert!(lhs.len() != 0);

        // NB (mat * row)^T == col * mat^T
        RowVec::new(mc(Transpose::Yes, rhs.0, (lhs.0).0, rhs.ncols()))
    }
}

impl<'a, 'b, T> Mul<&'a Mat<T>, RowVec<T>> for &'b RowVec<T> where T: Gemv + One + Zero {
    fn mul(self, rhs: &Mat<T>) -> RowVec<T> {
        self.as_row() * rhs.as_view()
    }
}

impl<'a, 'b, 'c, T> Mul<&'a MutView<'b, T>, RowVec<T>> for &'c RowVec<T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &MutView<T>) -> RowVec<T> {
        self.as_row() * rhs.as_view()
    }
}

impl<'a, 'b, T> Mul<&'a Trans<Mat<T>>, RowVec<T>> for &'b RowVec<T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &Trans<Mat<T>>) -> RowVec<T> {
        self.as_row() * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, 'c, T> Mul<&'a Trans<MutView<'b, T>>, RowVec<T>> for &'c RowVec<T> where
    T: Gemv + One + Zero,
{
    fn mul(self, rhs: &Trans<MutView<T>>) -> RowVec<T> {
        self.as_row() * Trans(rhs.0.as_view())
    }
}

impl<'a, 'b, T> Mul<Trans<View<'a, T>>, RowVec<T>> for &'b RowVec<T> where T: Gemv + One + Zero {
    fn mul(self, rhs: Trans<View<T>>) -> RowVec<T> {
        self.as_row() * rhs
    }
}

impl<'a, 'b, T> Mul<View<'a, T>, RowVec<T>> for &'b RowVec<T> where T: Gemv + One + Zero {
    fn mul(self, rhs: View<T>) -> RowVec<T> {
        self.as_row() * rhs
    }
}
