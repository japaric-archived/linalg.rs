use std::ops::{IndexAssign, Range, RangeFrom, RangeFull};

use extract::Extract;

use traits::Matrix;

// Single contiguous iterator pass (which ideally should optimize to a memset)
// NOTE Core
impl<'a, T, O> IndexAssign<RangeFull, T> for ::Mat<T, O> where T: Clone {
    fn index_assign(&mut self, _: RangeFull, rhs: T) {
        let src = rhs;

        for dst in self.as_mut().iter_mut() {
            *dst = src.clone();
        }
    }
}

// Single contiguous iterator pass (which ideally should optimize to a memcpy)
// NOTE Core
impl<'a, T, O> IndexAssign<RangeFull, &'a ::Mat<T, O>> for ::Mat<T, O> where T: Clone {
    fn index_assign(&mut self, _: RangeFull, rhs: &::Mat<T, O>) {
        unsafe {
            assert_eq!(self.size(), rhs.size());

            let mut src = rhs.as_ref().iter();

            for dst in self.as_mut().iter_mut() {
                *dst = src.next().extract().clone();
            }
        }
    }
}

// NOTE Secondary
impl<'a, T> IndexAssign<(RangeFull, Range<u32>), &'a ::Mat<T, ::order::Col>> for ::Mat<T, ::order::Col> where
    T: Clone,
{
    fn index_assign(&mut self, (r, c): (RangeFull, Range<u32>), rhs: &::Mat<T, ::order::Col>) {
        ::Mat::index_assign(&mut self[r, c], .., rhs)
    }
}

// NOTE Secondary
impl<'a, T> IndexAssign<(RangeFull, RangeFrom<u32>), &'a ::Mat<T, ::order::Col>> for ::Mat<T, ::order::Col> where
    T: Clone,
{
    fn index_assign(&mut self, (r, c): (RangeFull, RangeFrom<u32>), rhs: &::Mat<T, ::order::Col>) {
        ::Mat::index_assign(&mut self[r, c], .., rhs)
    }
}

// NOTE Secondary
impl<'a, T> IndexAssign<(RangeFull, u32), T> for ::strided::Mat<T, ::order::Col> where
    T: Clone,
{
    fn index_assign(&mut self, (r, c): (RangeFull, u32), rhs: T) {
        ::Col::index_assign(&mut self[r, c], .., rhs)
    }
}

// Two nested iterators where the inner one is contiguous (and ideally should optimize to a memcpy)
// NOTE Core
impl<'a, T> IndexAssign<RangeFull, &'a ::strided::Mat<T, ::order::Col>> for ::strided::Mat<T, ::order::Col> where
    T: Clone,
{
    fn index_assign(&mut self, _: RangeFull, rhs: &::strided::Mat<T, ::order::Col>) {
        unsafe {
            assert_eq!(self.size(), rhs.size());

            let mut src = rhs.cols();
            for dst in self.cols_mut() {
                dst[..] = src.next().extract();
            }
        }
    }
}

// NOTE Core
impl<'a, T> IndexAssign<RangeFull, &'a ::strided::Mat<T, ::order::Row>> for ::strided::Mat<T, ::order::Row> where
    T: Clone,
{
    fn index_assign(&mut self, _: RangeFull, rhs: &::strided::Mat<T, ::order::Row>) {
        unsafe {
            assert_eq!(self.size(), rhs.size());

            let mut src = rhs.rows();
            for dst in self.rows_mut() {
                dst[..] = src.next().extract();
            }
        }
    }
}

// NB All these "forward impls" shouldn't be necessary, but there is a limitation in the index
// overloading code. See rust-lang/rust#26218
// NOTE Secondary
impl<'a, T> IndexAssign<(RangeFull, u32), T> for ::Mat<T, ::order::Col> where
    T: Clone,
{
    fn index_assign(&mut self, (r, c): (RangeFull, u32), rhs: T) {
        ::strided::Mat::index_assign(&mut **self, (r, c), rhs)
    }
}
