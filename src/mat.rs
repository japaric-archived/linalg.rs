use std::borrow::{Cow, IntoCow};
use std::marker::PhantomData;
use std::num::{One, Zero};
use std::ops::{
    Deref, DerefMut, Index, IndexAssign, IndexMut, Range, RangeFrom, RangeFull, RangeTo,
};
use std::{fmt, iter, mem, slice};

use core::nonzero::NonZero;

use cast::From;
use extract::Extract;

use order::Order;
use traits::{Matrix, Transpose};
use u31::U31;

impl<T, O> ::Mat<T, O> {
    /// Returns an iterator over the elements of this matrix
    pub fn iter(&self) -> slice::Iter<T> {
        self.as_ref().iter()
    }

    /// Returns a mutable iterator over the elements of this matrix
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.as_mut().iter_mut()
    }

    /// Returns the raw representation of this matrix
    pub fn repr(&self) -> ::raw::Mat<T, O> {
        unsafe {
            mem::transmute(self)
        }
    }

    // NOTE Core
    fn as_slice_raw(&self) -> *mut [T] {
        unsafe {
            let ::raw::Mat { data, nrows, ncols, .. } = self.repr();

            slice::from_raw_parts_mut(*data, nrows.usize() * ncols.usize())
        }
    }
}

impl<T, O> ::Mat<T, O> where O: Order {
    /// Creates a matrix where each element is initialized to `elem`
    pub fn from_elem((nrows, ncols): (u32, u32), elem: T) -> Box<::Mat<T, O>> where T: Clone {
        unsafe {
            let n = usize::from(nrows).checked_mul(usize::from(ncols)).unwrap();
            let (nrows, ncols) = (U31::from(nrows).unwrap(), U31::from(ncols).unwrap());

            let elems = iter::repeat(elem).take(n).collect::<Vec<_>>().into_boxed_slice();

            ::Mat::from_raw_parts(elems, (nrows, ncols))
        }
    }

    /// Creates a matrix where each element is initialized using the function `f`
    pub fn from_fn<F>((nrows, ncols): (u32, u32), mut f: F) -> Box<::Mat<T, O>> where F: FnMut((u32, u32)) -> T {
        unsafe {
            let n = usize::from(nrows).checked_mul(usize::from(ncols)).unwrap();
            let (nrows_, ncols_) = (U31::from(nrows).unwrap(), U31::from(ncols).unwrap());

            let mut v = Vec::with_capacity(n);

            match O::order() {
                ::Order::Col => {
                    for col in 0..ncols {
                        for row in 0..nrows {
                            v.push(f((row, col)));
                        }
                    }
                },
                ::Order::Row => {
                    for row in 0..nrows {
                        for col in 0..ncols {
                            v.push(f((row, col)));
                        }
                    }
                },
            }
            ::Mat::from_raw_parts(v.into_boxed_slice(), (nrows_, ncols_))
        }
    }

    /// Reshapes a slice into a matrix
    pub fn reshape(slice: &[T], (nrows, ncols): (u32, u32)) -> &::Mat<T, O> {
        unsafe {
            &*::Mat::reshape_raw(slice, (nrows, ncols))
        }
    }

    /// Reshapes a mutable slice into a matrix
    pub fn reshape_mut(slice: &mut [T], (nrows, ncols): (u32, u32)) -> &mut ::Mat<T, O> {
        unsafe {
            &mut *::Mat::reshape_raw(slice, (nrows, ncols))
        }
    }

    /// Creates a matrix from an owned slice
    pub fn reshape_owned(elems: Box<[T]>, (nrows, ncols): (u32, u32)) -> Box<::Mat<T, O>> {
        unsafe {
            assert_eq!(elems.len(), usize::from(nrows) * usize::from(ncols));
            let nrows = U31::from(nrows).unwrap();
            let ncols = U31::from(ncols).unwrap();
            ::Mat::from_raw_parts(elems, (nrows, ncols))
        }
    }

    /// Creates a matrix filled with ones
    pub fn ones((nrows, ncols): (u32, u32)) -> Box<::Mat<T, O>> where T: Clone + One {
        ::Mat::from_elem((nrows, ncols), T::one())
    }

    /// Creates a matrix filled with zeros
    pub fn zeros((nrows, ncols): (u32, u32)) -> Box<::Mat<T, O>> where T: Clone + Zero {
        ::Mat::from_elem((nrows, ncols), T::zero())
    }

    // NOTE Core
    fn deref_raw(&self) -> *mut ::strided::Mat<T, O> {
        unsafe {
            let ::raw::Mat { data, nrows, ncols, marker } = self.repr();

            match O::order() {
                ::Order::Col => {
                    mem::transmute(::strided::raw::Mat {
                        data: data,
                        marker: marker,
                        ncols: ncols,
                        nrows: nrows,
                        stride: nrows,
                    })
                },
                ::Order::Row => {
                    mem::transmute(::strided::raw::Mat {
                        data: data,
                        marker: marker,
                        ncols: ncols,
                        nrows: nrows,
                        stride: ncols,
                    })
                },
            }
        }
    }

    unsafe fn from_raw_parts(mut elems: Box<[T]>, (nrows, ncols): (U31, U31)) -> Box<::Mat<T, O>> {
        let data = NonZero::new(elems.as_mut_ptr());
        mem::forget(elems);

        mem::transmute(::raw::Mat {
            data: data,
            nrows: nrows,
            ncols: ncols,
            marker: PhantomData::<O>,
        })
    }

    // NOTE Core
    fn reshape_raw(slice: &[T], (nrows, ncols): (u32, u32)) -> *mut ::Mat<T, O> {
        unsafe {
            assert_eq!(slice.len(), usize::from(nrows).checked_mul(usize::from(ncols)).unwrap());

            let ncols_ = U31::from(ncols).unwrap();
            let nrows_ = U31::from(nrows).unwrap();
            let data = slice.as_ptr() as *mut T;

            mem::transmute(::raw::Mat {
                data: NonZero::new(data),
                marker: PhantomData::<O>,
                ncols: ncols_,
                nrows: nrows_,
            })
        }
    }

    // NOTE Core
    fn t_raw(&self) -> *mut ::Mat<T, O::Transposed> {
        unsafe {
            let ::raw::Mat { data, nrows, ncols, .. } = self.repr();

            mem::transmute(::raw::Mat {
                data: data,
                marker: PhantomData::<O::Transposed>,
                ncols: nrows,
                nrows: ncols,
            })
        }
    }
}

impl<T> ::Mat<T, ::order::Col> {
    /// Returns an iterator over vertical stripes of this matrix
    pub fn vstripes(&self, size: u32) -> ::VStripes<T> {
        assert!(size != 0);

        ::VStripes {
            m: self,
            size: size,
        }
    }

    /// Vertically splits this matrix in two halves
    pub fn vsplit_at(&self, i: u32) -> (&::Mat<T, ::order::Col>, &::Mat<T, ::order::Col>) {
        unsafe {
            let (left, right) = self.vsplit_at_raw(i);
            (&*left, &*right)
        }
    }

    /// Vertically splits this matrix in two mutable halves
    pub fn vsplit_at_mut(&mut self, i: u32) -> (&mut ::Mat<T, ::order::Col>, &mut ::Mat<T, ::order::Col>) {
        unsafe {
            let (left, right) = self.vsplit_at_raw(i);
            (&mut *left, &mut *right)
        }
    }

    /// Returns a mutable iterator over vertical stripes of this matrix
    pub fn vstripes_mut(&mut self, size: u32) -> ::VStripesMut<T> {
        assert!(size != 0);

        ::VStripesMut {
            m: self,
            size: size,
        }
    }

    // NOTE Core
    fn col_slice_raw(&self, r: Range<u32>) -> *mut ::Mat<T, ::order::Col> {
        unsafe {
            let ::raw::Mat { data, nrows, ncols, marker } = self.repr();

            assert!(r.start <= r.end);
            assert!(r.end <= ncols.u32());

            mem::transmute(::raw::Mat {
                data: NonZero::new(data.offset(r.start as isize * nrows.isize())),
                marker: marker,
                ncols: U31::from(r.end - r.start).extract(),
                nrows: nrows,
            })
        }
    }

    // NOTE Core
    fn vsplit_at_raw(&self, i: u32) -> (*mut ::Mat<T, ::order::Col>, *mut ::Mat<T, ::order::Col>) {
        unsafe {
            assert!(i <= self.ncols());

            let ::raw::Mat { data, nrows, ncols, marker } = self.repr();
            let i = U31::from(i).extract();
            let j = ncols.checked_sub(i.i32()).extract();

            let left = mem::transmute(::raw::Mat {
                data: data,
                nrows: nrows,
                ncols: i,
                marker: marker,
            });
            let right = mem::transmute(::raw::Mat {
                data: NonZero::new(data.offset(i.isize() * nrows.isize())),
                nrows: nrows,
                ncols: j,
                marker: marker,
            });

            (left, right)
        }
    }
}

impl<T> ::Mat<T, ::order::Row> {
    /// Horizontally splits this matrix in two halves
    pub fn hsplit_at(&self, i: u32) -> (&::Mat<T, ::order::Row>, &::Mat<T, ::order::Row>) {
        unsafe {
            let (top, bottom) = self.hsplit_at_raw(i);
            (&*top, &*bottom)
        }
    }

    /// Horizontally splits this matrix in two mutable halves
    pub fn hsplit_at_mut(&mut self, i: u32) -> (&mut ::Mat<T, ::order::Row>, &mut ::Mat<T, ::order::Row>) {
        unsafe {
            let (top, bottom) = self.hsplit_at_raw(i);
            (&mut *top, &mut *bottom)
        }
    }

    /// Returns an iterator over horizontal stripes of this matrix
    pub fn hstripes(&self, size: u32) -> ::HStripes<T> {
        assert!(size != 0);

        ::HStripes {
            m: self,
            size: size,
        }
    }

    /// Returns a mutable iterator over horizontal stripes of this matrix
    pub fn hstripes_mut(&mut self, size: u32) -> ::HStripesMut<T> {
        assert!(size != 0);

        ::HStripesMut {
            m: self,
            size: size,
        }
    }

    // NOTE Core
    fn hsplit_at_raw(&self, i: u32) -> (*mut ::Mat<T, ::order::Row>, *mut ::Mat<T, ::order::Row>) {
        unsafe {
            assert!(i <= self.nrows());

            let ::raw::Mat { data, nrows, ncols, marker } = self.repr();
            let i = U31::from(i).extract();
            let j = nrows.checked_sub(i.i32()).extract();

            let top = mem::transmute(::raw::Mat {
                data: data,
                nrows: i,
                ncols: ncols,
                marker: marker,
            });
            let bottom = mem::transmute(::raw::Mat {
                data: NonZero::new(data.offset(i.isize() * ncols.isize())),
                nrows: j,
                ncols: ncols,
                marker: marker,
            });

            (top, bottom)
        }
    }

    // NOTE Core
    fn row_slice_raw(&self, r: Range<u32>) -> *mut ::Mat<T, ::order::Row> {
        unsafe {
            let ::raw::Mat { data, nrows, ncols, marker } = self.repr();

            assert!(r.start <= r.end);
            assert!(r.end <= nrows.u32());

            mem::transmute(::raw::Mat {
                data: NonZero::new(data.offset(r.start as isize * ncols.isize())),
                marker: marker,
                ncols: ncols,
                nrows: U31::from(r.end - r.start).extract(),
            })
        }
    }
}

impl<T, O> AsMut<[T]> for ::Mat<T, O> {
    fn as_mut(&mut self) -> &mut [T] {
        unsafe {
            &mut *self.as_slice_raw()
        }
    }
}

impl<T, O> AsRef<[T]> for ::Mat<T, O> {
    fn as_ref(&self) -> &[T] {
        unsafe {
            &*self.as_slice_raw()
        }
    }
}

impl<T> fmt::Debug for ::Mat<T, ::order::Col> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T> fmt::Debug for ::Mat<T, ::order::Row> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T, O> Deref for ::Mat<T, O> where O: Order {
    type Target = ::strided::Mat<T, O>;

    fn deref(&self) -> &::strided::Mat<T, O> {
        unsafe {
            &*self.deref_raw()
        }
    }
}

impl<T, O> DerefMut for ::Mat<T, O> where O: Order {
    fn deref_mut(&mut self) -> &mut ::strided::Mat<T, O> {
        unsafe {
            &mut *self.deref_raw()
        }
    }
}

impl<T, O> Drop for ::Mat<T, O> {
    fn drop(&mut self) {
        unsafe {
            let ::raw::Mat { data, nrows, ncols, .. } = self.repr();

            if !data.is_null() && *data as usize != mem::POST_DROP_USIZE {
                let len = nrows.usize() * ncols.usize();

                mem::drop(Vec::from_raw_parts(*data, len, len))
            }
        }
    }
}

/// Row slicing: `&mat[a..b, ..]`
impl<T> Index<(Range<u32>, RangeFull)> for ::Mat<T, ::order::Row> {
    type Output = ::Mat<T, ::order::Row>;

    fn index(&self, (r, _): (Range<u32>, RangeFull)) -> &::Mat<T, ::order::Row> {
        self.index(r)
    }
}

/// Row slicing: `&mat[a.., ..]`
impl<T> Index<(RangeFrom<u32>, RangeFull)> for ::Mat<T, ::order::Row> {
    type Output = ::Mat<T, ::order::Row>;

    fn index(&self, (r, _): (RangeFrom<u32>, RangeFull)) -> &::Mat<T, ::order::Row> {
        &self[r.start..self.nrows()]
    }
}

/// Row slicing: `&mat[..b, ..]`
impl<T> Index<(RangeTo<u32>, RangeFull)> for ::Mat<T, ::order::Row> {
    type Output = ::Mat<T, ::order::Row>;

    fn index(&self, (r, _): (RangeTo<u32>, RangeFull)) -> &::Mat<T, ::order::Row> {
        &self[0..r.end]
    }
}

/// Column slicing: `&mat[.., a..b]`
impl<T> Index<(RangeFull, Range<u32>)> for ::Mat<T, ::order::Col> {
    type Output = ::Mat<T, ::order::Col>;

    fn index(&self, (_, c): (RangeFull, Range<u32>)) -> &::Mat<T, ::order::Col> {
        unsafe {
            &*self.col_slice_raw(c)
        }
    }
}

/// Column slicing: `&mat[.., a..]`
impl<T> Index<(RangeFull, RangeFrom<u32>)> for ::Mat<T, ::order::Col> {
    type Output = ::Mat<T, ::order::Col>;

    fn index(&self, (r, c): (RangeFull, RangeFrom<u32>)) -> &::Mat<T, ::order::Col> {
        self.index((r, c.start..self.ncols()))
    }
}

/// Column slicing: `&mat[.., ..b]`
impl<T> Index<(RangeFull, RangeTo<u32>)> for ::Mat<T, ::order::Col> {
    type Output = ::Mat<T, ::order::Col>;

    fn index(&self, (r, c): (RangeFull, RangeTo<u32>)) -> &::Mat<T, ::order::Col> {
        self.index((r, 0..c.end))
    }
}

/// Row slicing: `&mat[a..b]`
impl<T> Index<Range<u32>> for ::Mat<T, ::order::Row> {
    type Output = ::Mat<T, ::order::Row>;

    fn index(&self, r: Range<u32>) -> &::Mat<T, ::order::Row> {
        unsafe {
            &*self.row_slice_raw(r)
        }
    }
}

/// Row slicing: `&mat[a..]`
impl<T> Index<RangeFrom<u32>> for ::Mat<T, ::order::Row> {
    type Output = ::Mat<T, ::order::Row>;

    fn index(&self, r: RangeFrom<u32>) -> &::Mat<T, ::order::Row> {
        &self[r.start..self.nrows()]
    }
}

/// Row slicing: `&mat[..b]`
impl<T> Index<RangeTo<u32>> for ::Mat<T, ::order::Row> {
    type Output = ::Mat<T, ::order::Row>;

    fn index(&self, r: RangeTo<u32>) -> &::Mat<T, ::order::Row> {
        &self[0..r.end]
    }
}

/// Mutable row slicing: `&mut mat[a..b, ..]`
impl<T> IndexMut<(Range<u32>, RangeFull)> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, (r, _): (Range<u32>, RangeFull)) -> &mut ::Mat<T, ::order::Row> {
        self.index_mut(r)
    }
}

/// Mutable row slicing: `&mut mat[a.., ..]`
impl<T> IndexMut<(RangeFrom<u32>, RangeFull)> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, (r, _): (RangeFrom<u32>, RangeFull)) -> &mut ::Mat<T, ::order::Row> {
        self.index_mut(r)
    }
}

/// Mutable row slicing: `&mut mat[..b, ..]`
impl<T> IndexMut<(RangeTo<u32>, RangeFull)> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, (r, _): (RangeTo<u32>, RangeFull)) -> &mut ::Mat<T, ::order::Row> {
        self.index_mut(r)
    }
}

/// Mutable column slicing: `&mut mat[.., a..b]`
impl<T> IndexMut<(RangeFull, Range<u32>)> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, (_, c): (RangeFull, Range<u32>)) -> &mut ::Mat<T, ::order::Col> {
        unsafe {
            &mut *self.col_slice_raw(c)
        }
    }
}

/// Mutable column slicing: `&mut mat[.., a..]`
impl<T> IndexMut<(RangeFull, RangeFrom<u32>)> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, (r, c): (RangeFull, RangeFrom<u32>)) -> &mut ::Mat<T, ::order::Col> {
        let end = self.ncols();
        self.index_mut((r, c.start..end))
    }
}

/// Mutable column slicing: `&mut mat[.., ..b]`
impl<T> IndexMut<(RangeFull, RangeTo<u32>)> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, (r, c): (RangeFull, RangeTo<u32>)) -> &mut ::Mat<T, ::order::Col> {
        self.index_mut((r, 0..c.end))
    }
}

/// Mutable row slicing: `&mut mat[a..b]`
impl<T> IndexMut<Range<u32>> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, r: Range<u32>) -> &mut ::Mat<T, ::order::Row> {
        unsafe {
            &mut *self.row_slice_raw(r)
        }
    }
}

/// Mutable row slicing: `&mut mat[a..]`
impl<T> IndexMut<RangeFrom<u32>> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, r: RangeFrom<u32>) -> &mut ::Mat<T, ::order::Row> {
        let end = self.nrows();
        &mut self[r.start..end]
    }
}

/// Mutable row slicing: `&mut mat[..b]`
impl<T> IndexMut<RangeTo<u32>> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, r: RangeTo<u32>) -> &mut ::Mat<T, ::order::Row> {
        &mut self[0..r.end]
    }
}

// NB All these "forward impls" shouldn't be necessary, but there is a limitation in the index
// overloading code. See rust-lang/rust#26218
/// Slicing: `&mat[a..b, c..d]`
impl<T, O> Index<(Range<u32>, Range<u32>)> for ::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (Range<u32>, Range<u32>)) -> &::strided::Mat<T, O> {
        self.deref().index((r, c))
    }
}

/// Slicing: `&mat[a..b, c..]`
impl<T, O> Index<(Range<u32>, RangeFrom<u32>)> for ::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (Range<u32>, RangeFrom<u32>)) -> &::strided::Mat<T, O> {
        self.deref().index((r, c))
    }
}

/// Slicing: `&mat[a..b, ..d]`
impl<T, O> Index<(Range<u32>, RangeTo<u32>)> for ::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (Range<u32>, RangeTo<u32>)) -> &::strided::Mat<T, O> {
        self.deref().index((r, c))
    }
}

/// Slicing: `&mat[a.., c..d]`
impl<T, O> Index<(RangeFrom<u32>, Range<u32>)> for ::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeFrom<u32>, Range<u32>)) -> &::strided::Mat<T, O> {
        self.deref().index((r, c))
    }
}

/// Slicing: `&mat[a.., c..]`
impl<T, O> Index<(RangeFrom<u32>, RangeFrom<u32>)> for ::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeFrom<u32>, RangeFrom<u32>)) -> &::strided::Mat<T, O> {
        self.deref().index((r, c))
    }
}

/// Slicing: `&mat[a.., ..d]`
impl<T, O> Index<(RangeFrom<u32>, RangeTo<u32>)> for ::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeFrom<u32>, RangeTo<u32>)) -> &::strided::Mat<T, O> {
        self.deref().index((r, c))
    }
}

/// Slicing: `&mat[..b, c..d]`
impl<T, O> Index<(RangeTo<u32>, Range<u32>)> for ::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeTo<u32>, Range<u32>)) -> &::strided::Mat<T, O> {
        self.deref().index((r, c))
    }
}

/// Slicing: `&mat[..b, c..]`
impl<T, O> Index<(RangeTo<u32>, RangeFrom<u32>)> for ::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeTo<u32>, RangeFrom<u32>)) -> &::strided::Mat<T, O> {
        self.deref().index((r, c))
    }
}

/// Slicing: `&mat[..b, ..d]`
impl<T, O> Index<(RangeTo<u32>, RangeTo<u32>)> for ::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeTo<u32>, RangeTo<u32>)) -> &::strided::Mat<T, O> {
        self.deref().index((r, c))
    }
}

/// Row slicing: `&mat[a..b, ..]`
impl<T> Index<(Range<u32>, RangeFull)> for ::Mat<T, ::order::Col> {
    type Output = ::strided::Mat<T, ::order::Col>;

    fn index(&self, (r, _): (Range<u32>, RangeFull)) -> &::strided::Mat<T, ::order::Col> {
        self.deref().index((r, ..))
    }
}

/// Row slicing: `&mat[a.., ..]`
impl<T> Index<(RangeFrom<u32>, RangeFull)> for ::Mat<T, ::order::Col> {
    type Output = ::strided::Mat<T, ::order::Col>;

    fn index(&self, (r, _): (RangeFrom<u32>, RangeFull)) -> &::strided::Mat<T, ::order::Col> {
        self.deref().index((r, ..))
    }
}

/// Row slicing: `&mat[..b, ..]`
impl<T> Index<(RangeTo<u32>, RangeFull)> for ::Mat<T, ::order::Col> {
    type Output = ::strided::Mat<T, ::order::Col>;

    fn index(&self, (r, _): (RangeTo<u32>, RangeFull)) -> &::strided::Mat<T, ::order::Col> {
        self.deref().index((r, ..))
    }
}

/// Column slicing: `&mat[.., a..b]`
impl<T> Index<(RangeFull, Range<u32>)> for ::Mat<T, ::order::Row> {
    type Output = ::strided::Mat<T, ::order::Row>;

    fn index(&self, (_, c): (RangeFull, Range<u32>)) -> &::strided::Mat<T, ::order::Row> {
        self.deref().index((.., c))
    }
}

/// Column slicing: `&mat[.., a..]`
impl<T> Index<(RangeFull, RangeFrom<u32>)> for ::Mat<T, ::order::Row> {
    type Output = ::strided::Mat<T, ::order::Row>;

    fn index(&self, (_, c): (RangeFull, RangeFrom<u32>)) -> &::strided::Mat<T, ::order::Row> {
        self.deref().index((.., c))
    }
}

/// Column slicing: `&mat[.., ..b]`
impl<T> Index<(RangeFull, RangeTo<u32>)> for ::Mat<T, ::order::Row> {
    type Output = ::strided::Mat<T, ::order::Row>;

    fn index(&self, (_, c): (RangeFull, RangeTo<u32>)) -> &::strided::Mat<T, ::order::Row> {
        self.deref().index((.., c))
    }
}

/// Column indexing: `&mat[.., i]`
impl<T> Index<(RangeFull, u32)> for ::Mat<T, ::order::Col> {
    type Output = ::Col<T>;

    fn index(&self, (_, i): (RangeFull, u32)) -> &::Col<T> {
        self.deref().index((.., i))
    }
}

/// Column indexing: `&mat[.., i]`
impl<T> Index<(RangeFull, u32)> for ::Mat<T, ::order::Row> {
    type Output = ::strided::Col<T>;

    fn index(&self, (_, i): (RangeFull, u32)) -> &::strided::Col<T> {
        self.deref().index((.., i))
    }
}

/// Row indexing: `&mat[i]`
impl<T> Index<u32> for ::Mat<T, ::order::Col> {
    type Output = ::strided::Row<T>;

    fn index(&self, i: u32) -> &::strided::Row<T> {
        self.deref().index(i)
    }
}

/// Row indexing: `&mat[i]`
impl<T> Index<u32> for ::Mat<T, ::order::Row> {
    type Output = ::Row<T>;

    fn index(&self, i: u32) -> &::Row<T> {
        self.deref().index(i)
    }
}

/// Element indexing: `&mat[i, j]`
impl<T, O> Index<(u32, u32)> for ::Mat<T, O> where O: Order {
    type Output = T;

    fn index(&self, (r, c): (u32, u32)) -> &T {
        self.deref().index((r, c))
    }
}

/// Row slicing: `&mat[a..b]`
impl<T> Index<Range<u32>> for ::Mat<T, ::order::Col> {
    type Output = ::strided::Mat<T, ::order::Col>;

    fn index(&self, r: Range<u32>) -> &::strided::Mat<T, ::order::Col> {
        self.deref().index(r)
    }
}

/// Row slicing: `&mat[a..]`
impl<T> Index<RangeFrom<u32>> for ::Mat<T, ::order::Col> {
    type Output = ::strided::Mat<T, ::order::Col>;

    fn index(&self, r: RangeFrom<u32>) -> &::strided::Mat<T, ::order::Col> {
        self.deref().index(r)
    }
}

/// Row slicing: `&mat[..b]`
impl<T> Index<RangeTo<u32>> for ::Mat<T, ::order::Col> {
    type Output = ::strided::Mat<T, ::order::Col>;

    fn index(&self, r: RangeTo<u32>) -> &::strided::Mat<T, ::order::Col> {
        self.deref().index(r)
    }
}

/// Setting an element: `mat[i, j] = x`
impl<T, O> IndexAssign<(u32, u32), T> for ::Mat<T, O> where O: Order {
    fn index_assign(&mut self, (i, j): (u32, u32), rhs: T) {
        self.deref_mut().index_assign((i, j), rhs)
    }
}

/// Mutable slicing: `&mut mat[a..b, c..d]`
impl<T, O> IndexMut<(Range<u32>, Range<u32>)> for ::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (Range<u32>, Range<u32>)) -> &mut ::strided::Mat<T, O> {
        self.deref_mut().index_mut((r, c))
    }
}

/// Mutable slicing: `&mut mat[a..b, c..]`
impl<T, O> IndexMut<(Range<u32>, RangeFrom<u32>)> for ::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (Range<u32>, RangeFrom<u32>)) -> &mut ::strided::Mat<T, O> {
        self.deref_mut().index_mut((r, c))
    }
}

/// Mutable slicing: `&mut mat[a..b, ..d]`
impl<T, O> IndexMut<(Range<u32>, RangeTo<u32>)> for ::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (Range<u32>, RangeTo<u32>)) -> &mut ::strided::Mat<T, O> {
        self.deref_mut().index_mut((r, c))
    }
}

/// Mutable slicing: `&mut mat[a.., c..d]`
impl<T, O> IndexMut<(RangeFrom<u32>, Range<u32>)> for ::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeFrom<u32>, Range<u32>)) -> &mut ::strided::Mat<T, O> {
        self.deref_mut().index_mut((r, c))
    }
}

/// Mutable slicing: `&mut mat[a.., c..]`
impl<T, O> IndexMut<(RangeFrom<u32>, RangeFrom<u32>)> for ::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeFrom<u32>, RangeFrom<u32>)) -> &mut ::strided::Mat<T, O> {
        self.deref_mut().index_mut((r, c))
    }
}

/// Mutable slicing: `&mut mat[a.., ..d]`
impl<T, O> IndexMut<(RangeFrom<u32>, RangeTo<u32>)> for ::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeFrom<u32>, RangeTo<u32>)) -> &mut ::strided::Mat<T, O> {
        self.deref_mut().index_mut((r, c))
    }
}

/// Mutable slicing: `&mut mat[..b, c..d]`
impl<T, O> IndexMut<(RangeTo<u32>, Range<u32>)> for ::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeTo<u32>, Range<u32>)) -> &mut ::strided::Mat<T, O> {
        self.deref_mut().index_mut((r, c))
    }
}

/// Mutable slicing: `&mut mat[..b, c..]`
impl<T, O> IndexMut<(RangeTo<u32>, RangeFrom<u32>)> for ::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeTo<u32>, RangeFrom<u32>)) -> &mut ::strided::Mat<T, O> {
        self.deref_mut().index_mut((r, c))
    }
}

/// Mutable slicing: `&mut mat[..b, ..d]`
impl<T, O> IndexMut<(RangeTo<u32>, RangeTo<u32>)> for ::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeTo<u32>, RangeTo<u32>)) -> &mut ::strided::Mat<T, O> {
        self.deref_mut().index_mut((r, c))
    }
}

/// Mutable row slicing: `&mut mat[a..b, ..]`
impl<T> IndexMut<(Range<u32>, RangeFull)> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, (r, _): (Range<u32>, RangeFull)) -> &mut ::strided::Mat<T, ::order::Col> {
        self.deref_mut().index_mut((r, ..))
    }
}

/// Mutable row slicing: `&mut mat[a.., ..]`
impl<T> IndexMut<(RangeFrom<u32>, RangeFull)> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, (r, _): (RangeFrom<u32>, RangeFull)) -> &mut ::strided::Mat<T, ::order::Col> {
        self.deref_mut().index_mut((r, ..))
    }
}

/// Mutable row slicing: `&mut mat[..b, ..]`
impl<T> IndexMut<(RangeTo<u32>, RangeFull)> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, (r, _): (RangeTo<u32>, RangeFull)) -> &mut ::strided::Mat<T, ::order::Col> {
        self.deref_mut().index_mut((r, ..))
    }
}

/// Mutable column slicing: `&mut mat[.., a..b]`
impl<T> IndexMut<(RangeFull, Range<u32>)> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, (_, c): (RangeFull, Range<u32>)) -> &mut ::strided::Mat<T, ::order::Row> {
        self.deref_mut().index_mut((.., c))
    }
}

/// Mutable column slicing: `&mut mat[.., a..]`
impl<T> IndexMut<(RangeFull, RangeFrom<u32>)> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, (_, c): (RangeFull, RangeFrom<u32>)) -> &mut ::strided::Mat<T, ::order::Row> {
        self.deref_mut().index_mut((.., c))
    }
}

/// Mutable column slicing: `&mut mat[.., ..b]`
impl<T> IndexMut<(RangeFull, RangeTo<u32>)> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, (_, c): (RangeFull, RangeTo<u32>)) -> &mut ::strided::Mat<T, ::order::Row> {
        self.deref_mut().index_mut((.., c))
    }
}

/// Mutable column indexing: `&mut mat[.., i]`
impl<T> IndexMut<(RangeFull, u32)> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, (_, i): (RangeFull, u32)) -> &mut ::Col<T> {
        self.deref_mut().index_mut((.., i))
    }
}

/// Mutable column indexing: `&mut mat[.., i]`
impl<T> IndexMut<(RangeFull, u32)> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, (_, i): (RangeFull, u32)) -> &mut ::strided::Col<T> {
        self.deref_mut().index_mut((.., i))
    }
}

/// Mutable row indexing: `&mut mat[i]`
impl<T> IndexMut<u32> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, i: u32) -> &mut ::strided::Row<T> {
        self.deref_mut().index_mut(i)
    }
}

/// Mutable row indexing: `&mut mat[i]`
impl<T> IndexMut<u32> for ::Mat<T, ::order::Row> {
    fn index_mut(&mut self, i: u32) -> &mut ::Row<T> {
        self.deref_mut().index_mut(i)
    }
}

/// Mutable element indexing: `&mut mat[i, j]`
impl<T, O> IndexMut<(u32, u32)> for ::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (u32, u32)) -> &mut T {
        self.deref_mut().index_mut((r, c))
    }
}

/// Mutable row slicing: `&mut mat[a..b]`
impl<T> IndexMut<Range<u32>> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, r: Range<u32>) -> &mut ::strided::Mat<T, ::order::Col> {
        self.deref_mut().index_mut(r)
    }
}

/// Mutable row slicing: `&mut mat[a..]`
impl<T> IndexMut<RangeFrom<u32>> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, r: RangeFrom<u32>) -> &mut ::strided::Mat<T, ::order::Col> {
        self.deref_mut().index_mut(r)
    }
}

/// Mutable row slicing: `&mut mat[..b]`
impl<T> IndexMut<RangeTo<u32>> for ::Mat<T, ::order::Col> {
    fn index_mut(&mut self, r: RangeTo<u32>) -> &mut ::strided::Mat<T, ::order::Col> {
        self.deref_mut().index_mut(r)
    }
}

impl<'a, T, O> IntoCow<'a, ::Mat<T, O>> for &'a ::Mat<T, O> where T: Clone {
    fn into_cow(self) -> Cow<'a, ::Mat<T, O>> {
        Cow::Borrowed(self)
    }
}

impl<'a, T, O> IntoCow<'a, ::Mat<T, O>> for Box<::Mat<T, O>> where T: Clone {
    fn into_cow(self) -> Cow<'a, ::Mat<T, O>> {
        Cow::Owned(self)
    }
}

impl<'a, T, O> IntoIterator for &'a ::Mat<T, O> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T, O> IntoIterator for &'a mut ::Mat<T, O> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T, O> Matrix for ::Mat<T, O> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        self.repr().nrows.u32()
    }

    fn ncols(&self) -> u32 {
        self.repr().ncols.u32()
    }
}

unsafe impl<T: Send, O> Send for ::Mat<T, O> {}

unsafe impl<T: Sync, O> Sync for ::Mat<T, O> {}

impl<T, O> ToOwned for ::Mat<T, O> where T: Clone {
    type Owned = Box<::Mat<T, O>>;

    fn to_owned(&self) -> Box<::Mat<T, O>> {
        unsafe {
            let ::raw::Mat { nrows, ncols, marker, .. } = self.repr();
            let mut v = self.as_ref().to_owned();
            let data = v.as_mut_ptr();
            mem::forget(v);
            mem::transmute(::raw::Mat {
                data: NonZero::new(data),
                marker: marker,
                ncols: ncols,
                nrows: nrows,
            })
        }

    }
}

impl<'a, T, O> Transpose for &'a ::Mat<T, O> where O: Order {
    type Output = &'a ::Mat<T, O::Transposed>;

    fn t(self) -> &'a ::Mat<T, O::Transposed> {
        unsafe {
            &*self.t_raw()
        }
    }
}

impl<'a, T, O> Transpose for &'a mut ::Mat<T, O> where O: Order {
    type Output = &'a mut ::Mat<T, O::Transposed>;

    fn t(self) -> &'a mut ::Mat<T, O::Transposed> {
        unsafe {
            &mut *self.t_raw()
        }
    }
}

impl<'a, T, O> Transpose for Box<::Mat<T, O>> where O: Order {
    type Output = Box<::Mat<T, O::Transposed>>;

    fn t(self) -> Box<::Mat<T, O::Transposed>> {
        unsafe {
            let t = self.t_raw();
            mem::forget(self);
            mem::transmute(t)
        }
    }
}
