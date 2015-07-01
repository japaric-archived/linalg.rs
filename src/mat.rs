use std::borrow::{Cow, IntoCow};
use std::marker::{PhantomData, Unsized};
use std::num::{One, Zero};
use std::ops::{
    Deref, DerefMut, Index, IndexAssign, IndexMut, Range, RangeFrom, RangeFull, RangeTo,
};
use std::raw::FatPtr;
use std::{fat_ptr, fmt, iter, mem, ptr, slice};

use cast::From;
use extract::Extract;

use order::Order;
use traits::{Matrix, Transpose};
use u31::U31;

pub struct Info<O> {
    pub ncols: U31,
    pub nrows: U31,
    pub _marker: PhantomData<O>,
}

impl<O> Clone for Info<O> {
    fn clone(&self) -> Info<O> {
        *self
    }
}

impl<O> Copy for Info<O> {}

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
    pub fn repr(&self) -> FatPtr<T, Info<O>> {
        fat_ptr::repr(self)
    }

    // NOTE Core
    fn as_slice_raw(&self) -> *mut [T] {
        let FatPtr { data, info } = self.repr();

        unsafe {
            slice::from_raw_parts_mut(data, info.nrows.usize() * info.ncols.usize())
        }
    }
}

impl<T, O> ::Mat<T, O> where O: Order {
    /// Creates a matrix where each element is initialized to `elem`
    pub fn from_elem((nrows, ncols): (u32, u32), elem: T) -> Box<::Mat<T, O>> where T: Clone {
        let (nrows, ncols) = (U31::from(nrows).unwrap(), U31::from(ncols).unwrap());
        let n = nrows.usize().checked_mul(ncols.usize()).unwrap();

        let mut v: Vec<_> = iter::repeat(elem).take(n).collect();
        let data = v.as_mut_ptr();
        mem::forget(v);

        unsafe {
            ::Mat::from_raw_parts(data, (nrows, ncols))
        }
    }

    /// Creates a matrix where each element is initialized using the function `f`
    pub fn from_fn<F>((nrows, ncols): (u32, u32), mut f: F) -> Box<::Mat<T, O>> where
        F: FnMut((u32, u32)) -> T,
    {
        let (nrows, ncols) = (U31::from(nrows).unwrap(), U31::from(ncols).unwrap());
        let n = nrows.usize().checked_mul(ncols.usize()).unwrap();

        let mut v = Vec::with_capacity(n);

        match O::order() {
            ::Order::Col => {
                for col in 0..ncols.u32() {
                    for row in 0..nrows.u32() {
                        v.push(f((row, col)));
                    }
                }
            },
            ::Order::Row => {
                for row in 0..nrows.u32() {
                    for col in 0..ncols.u32() {
                        v.push(f((row, col)));
                    }
                }
            },
        }

        let data = v.as_mut_ptr();
        mem::forget(v);

        unsafe {
            ::Mat::from_raw_parts(data, (nrows, ncols))
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
    pub fn reshape_owned(mut elems: Box<[T]>, (nrows, ncols): (u32, u32)) -> Box<::Mat<T, O>> {
        let (nrows, ncols) = (U31::from(nrows).unwrap(), U31::from(ncols).unwrap());

        assert_eq!(Some(elems.len()), nrows.usize().checked_mul(ncols.usize()));

        let data = elems.as_mut_ptr();
        mem::forget(elems);

        unsafe {
            ::Mat::from_raw_parts(data, (nrows, ncols))
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

    unsafe fn from_raw_parts(data: *mut T, (nrows, ncols): (U31, U31)) -> Box<::Mat<T, O>> {
        Box::from_raw(fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: PhantomData,
                ncols: ncols,
                nrows: nrows,
            }
        }))
    }

    // NOTE Core
    fn deref_raw(&self) -> *mut ::strided::Mat<T, O> {
        let FatPtr { data, info } = self.repr();

        fat_ptr::new(FatPtr {
            data: data,
            info: ::strided::mat::Info {
                _marker: info._marker,
                ncols: info.ncols,
                nrows: info.nrows,
                stride: match O::order() {
                    ::Order::Col => info.nrows,
                    ::Order::Row => info.ncols,
                }
            }
        })
    }

    // NOTE Core
    fn reshape_raw(slice: &[T], (nrows, ncols): (u32, u32)) -> *mut ::Mat<T, O> {
        let (nrows, ncols) = (U31::from(nrows).unwrap(), U31::from(ncols).unwrap());

        assert_eq!(Some(slice.len()), nrows.usize().checked_mul(ncols.usize()));

        fat_ptr::new(FatPtr {
            data: slice.as_ptr() as *mut T,
            info: Info {
                nrows: nrows,
                ncols: ncols,
                _marker: PhantomData,
            }
        })
    }

    // NOTE Core
    fn t_raw(&self) -> *mut ::Mat<T, O::Transposed> {
        let FatPtr { data, info } = self.repr();

        fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                nrows: info.ncols,
                ncols: info.nrows,
                _marker: PhantomData,
            }
        })
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
    unsafe fn col_slice_raw(&self, Range { start, end }: Range<u32>) -> *mut ::Mat<T, ::order::Col> {
        let FatPtr { data, info } = self.repr();

        assert!(start <= end);
        assert!(end <= info.ncols.u32());

        fat_ptr::new(FatPtr {
            data: data.offset(start as isize * info.nrows.isize()),
            info: Info {
                _marker: info._marker,
                ncols: U31::from(end - start).extract(),
                nrows: info.nrows,
            }
        })
    }

    // NOTE Core
    unsafe fn vsplit_at_raw(&self, i: u32) -> (*mut ::Mat<T, ::order::Col>, *mut ::Mat<T, ::order::Col>) {
        let FatPtr { data, info } = self.repr();

        assert!(i <= info.ncols.u32());

        let i = U31::from(i).extract();
        let j = info.ncols.checked_sub(i.i32()).extract();

        let left = fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: info._marker,
                ncols: i,
                nrows: info.nrows,
            }
        });
        let right = fat_ptr::new(FatPtr {
            data: data.offset(i.isize() * info.nrows.isize()),
            info: Info {
                nrows: info.nrows,
                ncols: j,
                _marker: info._marker,
            }
        });

        (left, right)
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
    unsafe fn hsplit_at_raw(&self, i: u32) -> (*mut ::Mat<T, ::order::Row>, *mut ::Mat<T, ::order::Row>) {
        let FatPtr { data, info } = self.repr();

        assert!(i <= info.nrows.u32());

        let i = U31::from(i).extract();
        let j = info.nrows.checked_sub(i.i32()).extract();

        let top = fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: info._marker,
                ncols: info.ncols,
                nrows: i,
            }
        });
        let bottom = fat_ptr::new(FatPtr {
            data: data.offset(i.isize() * info.ncols.isize()),
            info: Info {
                _marker: info._marker,
                ncols: info.ncols,
                nrows: j,
            }
        });

        (top, bottom)
    }

    // NOTE Core
    unsafe fn row_slice_raw(&self, Range { start, end }: Range<u32>) -> *mut ::Mat<T, ::order::Row> {
        let FatPtr { data, info } = self.repr();

        assert!(start <= end);
        assert!(end <= info.nrows.u32());

        fat_ptr::new(FatPtr {
            data: data.offset(start as isize * info.ncols.isize()),
            info: Info {
                _marker: info._marker,
                ncols: info.ncols,
                nrows: U31::from(end - start).extract(),
            },
        })
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
            let ptr = self.repr().data;

            if !ptr.is_null() && ptr as usize != mem::POST_DROP_USIZE {
                for x in &*self {
                    ptr::read(x);
                }
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
        self.repr().info.nrows.u32()
    }

    fn ncols(&self) -> u32 {
        self.repr().info.ncols.u32()
    }
}

unsafe impl<T: Send, O> Send for ::Mat<T, O> {}

unsafe impl<T: Sync, O> Sync for ::Mat<T, O> {}

impl<T, O> ToOwned for ::Mat<T, O> where T: Clone {
    type Owned = Box<::Mat<T, O>>;

    fn to_owned(&self) -> Box<::Mat<T, O>> {
        let FatPtr { info, .. } = self.repr();
        let mut v = self.as_ref().to_owned();
        let data = v.as_mut_ptr();
        mem::forget(v);

        unsafe {
            Box::from_raw(fat_ptr::new(FatPtr {
                data: data,
                info: info,
            }))
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

impl<T, O> Unsized for ::Mat<T, O> {
    type Data = T;
    type Info = Info<O>;

    fn size_of_val(info: Info<O>) -> usize {
        info.nrows.usize() * info.ncols.usize() * mem::size_of::<T>()
    }
}
