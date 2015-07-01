//! Iterators over strided matrices

use std::borrow::{Borrow, Cow, IntoCow};
use std::marker::{PhantomData, Unsized};
use std::num::Zero;
use std::ops::{Index, IndexAssign, IndexMut, Range, RangeFrom, RangeTo, RangeFull};
use std::raw::FatPtr;
use std::{cmp, fat_ptr, fmt, mem, slice};

use cast::From;
use extract::Extract;

use order::Order;
use traits::{Matrix, Transpose};
use u31::U31;

mod iter;

/// Extra information for strided matrices
#[allow(missing_docs)]
pub struct Info<Order> {
    /// Marker to "use" the `Order` of the matrix
    pub _marker: PhantomData<Order>,
    /// Number of columns
    pub ncols: U31,
    /// Number of rows
    pub nrows: U31,
    /// Stride of the matrix
    pub stride: U31,
}

impl<O> Clone for Info<O> {
    fn clone(&self) -> Info<O> {
        *self
    }
}

impl<O> Copy for Info<O> {}

/// Iterator over a strided matrix
pub struct Iter<'a, T: 'a, O: 'a> {
    m: &'a ::strided::Mat<T, O>,
    i: U31,
}

/// Iterator over a mutable strided matrix
pub struct IterMut<'a, T:'a , O:'a > {
    m: &'a mut ::strided::Mat<T, O>,
    i: U31,
}

impl<T, O> ::strided::Mat<T, O> {
    /// Returns an iterator over the columns of this matrix
    pub fn cols(&self) -> ::Cols<T, O> {
        ::Cols {
            m: self,
        }
    }

    /// Returns a mutable iterator over the columns of this matrix
    pub fn cols_mut(&mut self) -> ::ColsMut<T, O> {
        ::ColsMut {
            m: self,
        }
    }

    /// Returns an iterator over horizontal stripes of this matrix
    pub fn hstripes(&self, size: u32) -> ::strided::HStripes<T, O> {
        assert!(size != 0);

        ::strided::HStripes {
            m: self,
            size: size,
        }
    }

    /// Returns a mutable iterator over horizontal stripes of this matrix
    pub fn hstripes_mut(&mut self, size: u32) -> ::strided::HStripesMut<T, O> {
        assert!(size != 0);

        ::strided::HStripesMut {
            m: self,
            size: size,
        }
    }

    /// Returns an iterator over the elements of this matrix
    pub fn iter(&self) -> Iter<T, O> {
        Iter {
            m: if self.is_empty() { ::strided::Mat::empty() } else { self },
            i: U31::zero(),
        }
    }

    /// Returns a mutable iterator over the elements of this matrix
    pub fn iter_mut(&mut self) -> IterMut<T, O> {
        IterMut {
            m: if self.is_empty() { ::strided::Mat::empty() } else { self },
            i: U31::zero(),
        }
    }

    /// Returns the raw representation of this matrix
    pub fn repr(&self) -> FatPtr<T, ::strided::mat::Info<O>> {
        fat_ptr::repr(self)
    }

    /// Returns an iterator over the rows of this matrix
    pub fn rows(&self) -> ::Rows<T, O> {
        ::Rows {
            m: self,
        }
    }

    /// Returns a mutable iterator over the rows of this matrix
    pub fn rows_mut(&mut self) -> ::RowsMut<T, O> {
        ::RowsMut {
            m: self,
        }
    }

    /// Returns an iterator over vertical stripes of this matrix
    pub fn vstripes(&self, size: u32) -> ::strided::VStripes<T, O> {
        assert!(size != 0);

        ::strided::VStripes {
            m: self,
            size: size,
        }
    }

    /// Returns a mutable iterator over vertical stripes of this matrix
    pub fn vstripes_mut(&mut self, size: u32) -> ::strided::VStripesMut<T, O> {
        assert!(size != 0);

        ::strided::VStripesMut {
            m: self,
            size: size,
        }
    }
}

impl<T, O> ::strided::Mat<T, O> where O: Order {
    /// Returns a view into the `i`th diagonal of this matrix
    pub fn diag(&self, i: i32) -> &::strided::Diag<T> {
        unsafe {
            &*self.diag_raw(i)
        }
    }

    /// Returns a mutable view into the `i`th diagonal of this matrix
    pub fn diag_mut(&mut self, i: i32) -> &mut ::strided::Diag<T> {
        unsafe {
            &mut *self.diag_raw(i)
        }
    }

    /// Horizontally splits this matrix in two halves
    pub fn hsplit_at(&self, i: u32) -> (&::strided::Mat<T, O>, &::strided::Mat<T, O>) {
        unsafe {
            let (top, bottom) = self.hsplit_at_raw(i);
            (&*top, &*bottom)
        }
    }

    /// Horizontally splits this matrix in two mutable halves
    pub fn hsplit_at_mut(&mut self, i: u32) -> (&mut ::strided::Mat<T, O>, &mut ::strided::Mat<T, O>) {
        unsafe {
            let (top, bottom) = self.hsplit_at_raw(i);
            (&mut *top, &mut *bottom)
        }
    }

    /// Vertically splits this matrix in two halves
    pub fn vsplit_at(&self, i: u32) -> (&::strided::Mat<T, O>, &::strided::Mat<T, O>) {
        unsafe {
            let (left, right) = self.vsplit_at_raw(i);
            (&*left, &*right)
        }
    }

    /// Vertically splits this matrix in two mutable halves
    pub fn vsplit_at_mut(&mut self, i: u32) -> (&mut ::strided::Mat<T, O>, &mut ::strided::Mat<T, O>) {
        unsafe {
            let (left, right) = self.vsplit_at_raw(i);
            (&mut *left, &mut *right)
        }
    }

    fn as_slice(&self) -> Option<&[T]> {
        let FatPtr { data, info } = self.repr();

        if info.stride == match O::order() {
            ::Order::Col => info.nrows,
            ::Order::Row => info.ncols,
        } {
            let len = info.nrows.usize() * info.ncols.usize();

            Some(unsafe {
                slice::from_raw_parts(data, len)
            })
        } else {
            None
        }
    }

    // NOTE Core
    unsafe fn col_slice_raw(&self, Range { start, end }: Range<u32>) -> *mut ::strided::Mat<T, O> {
        let FatPtr { data, info } = self.repr();

        assert!(start <= end);
        assert!(end <= info.ncols.u32());

        let data = match O::order() {
            ::Order::Col => data.offset(start as isize * info.stride.isize()),
            ::Order::Row => data.offset(start as isize),
        };

        fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: info._marker,
                ncols: U31::from(end - start).extract(),
                nrows: info.nrows,
                stride: info.stride,
            }
        })
    }

    // NOTE Core
    unsafe fn diag_raw(&self, i: i32) -> *mut ::strided::Diag<T> {
        let FatPtr { data, info } = self.repr();

        // NB I'll work in column-major order space, if the input matrix is in row-major order,
        // then treat it as a transpose. Remember that `m.diag(i) === m.t().diag(-i)`
        let (nrows, ncols, i) = match O::order() {
            ::Order::Col => (info.nrows, info.ncols, i),
            ::Order::Row => (info.ncols, info.nrows, -i),
        };

        let v: *mut ::strided::Vector<T> = if i > 0 {
            let len = cmp::min(nrows, ncols.checked_sub(i).unwrap());
            let data = data.offset(isize::from(i) * info.stride.isize());
            let stride = info.stride + 1;

            fat_ptr::new(FatPtr {
                data: data,
                info: ::strided::vector::Info {
                    len: len,
                    stride: stride,
                }
            })
        } else {
            let i = -i;

            let len = cmp::min(nrows.checked_sub(i).unwrap(), ncols);
            let data = data.offset(isize::from(i));
            let stride = info.stride + 1;

            fat_ptr::new(FatPtr {
                data: data,
                info: ::strided::vector::Info {
                    len: len,
                    stride: stride,
                }
            })
        };

        mem::transmute(v)
    }

    // NOTE Core
    unsafe fn index_raw(&self, (row, col): (u32, u32)) -> *mut T {
        let FatPtr { data, info } = self.repr();

        assert!(row < info.nrows.u32());
        assert!(col < info.ncols.u32());

        match O::order() {
            ::Order::Col => data.offset(row as isize + info.stride.isize() * col as isize),
            ::Order::Row => data.offset(col as isize + info.stride.isize() * row as isize),
        }
    }

    // NOTE Core
    unsafe fn row_slice_raw(&self, Range { start, end }: Range<u32>) -> *mut ::strided::Mat<T, O> {
        let FatPtr { data, info } = self.repr();

        assert!(start <= end);
        assert!(end <= info.nrows.u32());

        let data = match O::order() {
            ::Order::Col => data.offset(start as isize),
            ::Order::Row => data.offset(start as isize * info.stride.isize()),
        };

        fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: info._marker,
                ncols: info.ncols,
                nrows: U31::from(end - start).extract(),
                stride: info.stride,
            }
        })
    }

    // NOTE Core
    unsafe fn slice_raw(&self, (r, c): (Range<u32>, Range<u32>)) -> *mut ::strided::Mat<T, O> {
        let FatPtr { data, info } = self.repr();

        assert!(r.start <= r.end);
        assert!(r.end <= info.nrows.u32());
        assert!(c.start <= c.end);
        assert!(c.end <= info.ncols.u32());

        let data = match O::order() {
            ::Order::Col => data.offset(r.start as isize + c.start as isize * info.stride.isize()),
            ::Order::Row => data.offset(c.start as isize + r.start as isize * info.stride.isize()),
        };

        fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: info._marker,
                ncols: U31::from(c.end - c.start).extract(),
                nrows: U31::from(r.end - r.start).extract(),
                stride: info.stride,
            }
        })
    }

    // NOTE Core
    fn hsplit_at_raw(&self, i: u32) -> (*mut ::strided::Mat<T, O>, *mut ::strided::Mat<T, O>) {
        let i = U31::from(i).unwrap();
        let FatPtr { data, info  } = self.repr();
        let j = info.nrows.checked_sub(i.i32()).unwrap();

        let top = fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: info._marker,
                ncols: info.ncols,
                nrows: i,
                stride: info.stride,
            }
        });

        let data = unsafe {
            match O::order() {
                ::Order::Col => data.offset(i.isize()),
                ::Order::Row => data.offset(i.isize() * info.stride.isize()),
            }
        };

        let bottom = fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: info._marker,
                ncols: info.ncols,
                nrows: j,
                stride: info.stride,
            }
        });

        (top, bottom)
    }

    // NOTE Core
    fn vsplit_at_raw(&self, i: u32) -> (*mut ::strided::Mat<T, O>, *mut ::strided::Mat<T, O>) {
        let i = U31::from(i).unwrap();
        let FatPtr { data, info } = self.repr();
        let j = info.ncols.checked_sub(i.i32()).unwrap();

        let left = fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: info._marker,
                ncols: i,
                nrows: info.nrows,
                stride: info.stride,
            }
        });

        let data = unsafe {
            match O::order() {
                ::Order::Col => data.offset(i.isize() * info.stride.isize()),
                ::Order::Row => data.offset(i.isize()),
            }
        };

        let right = fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: info._marker,
                ncols: j,
                nrows: info.nrows,
                stride: info.stride,
            }
        });

        (left, right)
    }

    // NOTE Core
    fn t_raw(&self) -> *mut ::strided::Mat<T, O::Transposed> {
        let FatPtr { data, info } = self.repr();

        fat_ptr::new(FatPtr {
            data: data,
            info: Info {
                _marker: PhantomData,
                ncols: info.nrows,
                nrows: info.ncols,
                stride: info.stride,
            }
        })
    }
}

impl<T> ::strided::Mat<T, ::order::Col> {
    // NOTE Core
    unsafe fn col_raw(&self, i: u32) -> *mut ::Col<T> {
        let FatPtr { data, info } = self.repr();

        assert!(i < info.ncols.u32());

        let v: *mut ::Vector<T> = fat_ptr::new(FatPtr {
            data: data.offset(i as isize * info.stride.isize()),
            info: info.nrows,
        });

        mem::transmute(v)
    }

    // NOTE Core
    unsafe fn strided_row_raw(&self, i: u32) -> *mut ::strided::Row<T> {
        let FatPtr { data, info } = self.repr();

        assert!(i < info.nrows.u32());

        let v: *mut ::strided::Vector<T> = fat_ptr::new(FatPtr {
            data: data.offset(i as isize),
            info: ::strided::vector::Info {
                len: info.ncols,
                stride: info.stride,
            }
        });

        mem::transmute(v)
    }
}

impl<T> ::strided::Mat<T, ::order::Row> {
    // NOTE Secondary
    fn row_raw(&self, i: u32) -> *mut ::Row<T> {
        unsafe {
            (&mut *self.t().col_raw(i)).t()
        }
    }

    // NOTE Secondary
    fn strided_col_raw(&self, i: u32) -> *mut ::strided::Col<T> {
        unsafe {
            (&mut *self.t().strided_row_raw(i)).t()
        }
    }
}

impl<T, O> Borrow<::strided::Mat<T, O>> for Box<::Mat<T, O>> where O: Order {
    fn borrow(&self) -> &::strided::Mat<T, O> {
        self
    }
}

impl<T> fmt::Debug for ::strided::Mat<T, ::order::Col> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut is_first = true;
        for row in self.rows() {
            if is_first {
                is_first = false;
            } else {
                try!(f.write_str("\n"))
            }

            try!(::strided::Vector::fmt(row, f))
        }

        Ok(())
    }
}

impl<T> fmt::Debug for ::strided::Mat<T, ::order::Row> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut is_first = true;
        for row in self.rows() {
            if is_first {
                is_first = false;
            } else {
                try!(f.write_str("\n"))
            }

            try!(::strided::Vector::fmt(row, f))
        }

        Ok(())
    }
}

/// Slicing: `&mat[a..b, c..d]`
impl<T, O> Index<(Range<u32>, Range<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (Range<u32>, Range<u32>)) -> &::strided::Mat<T, O> {
        unsafe {
            &*self.slice_raw((r, c))
        }
    }
}

/// Slicing: `&mat[a..b, c..]`
impl<T, O> Index<(Range<u32>, RangeFrom<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (Range<u32>, RangeFrom<u32>)) -> &::strided::Mat<T, O> {
        self.index((r, c.start..self.ncols()))
    }
}

/// Slicing: `&mat[a..b, ..d]`
impl<T, O> Index<(Range<u32>, RangeTo<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (Range<u32>, RangeTo<u32>)) -> &::strided::Mat<T, O> {
        self.index((r, 0..c.end))
    }
}

/// Slicing: `&mat[a.., c..d]`
impl<T, O> Index<(RangeFrom<u32>, Range<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeFrom<u32>, Range<u32>)) -> &::strided::Mat<T, O> {
        self.index((r.start..self.nrows(), c))
    }
}

/// Slicing: `&mat[a.., c..]`
impl<T, O> Index<(RangeFrom<u32>, RangeFrom<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeFrom<u32>, RangeFrom<u32>)) -> &::strided::Mat<T, O> {
        self.index((r.start..self.nrows(), c.start..self.ncols()))
    }
}

/// Slicing: `&mat[a.., ..d]`
impl<T, O> Index<(RangeFrom<u32>, RangeTo<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeFrom<u32>, RangeTo<u32>)) -> &::strided::Mat<T, O> {
        self.index((r.start..self.nrows(), 0..c.end))
    }
}

/// Slicing: `&mat[..b, c..d]`
impl<T, O> Index<(RangeTo<u32>, Range<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeTo<u32>, Range<u32>)) -> &::strided::Mat<T, O> {
        self.index((0..r.end, c))
    }
}

/// Slicing: `&mat[..b, c..]`
impl<T, O> Index<(RangeTo<u32>, RangeFrom<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeTo<u32>, RangeFrom<u32>)) -> &::strided::Mat<T, O> {
        self.index((0..r.end, c.start..self.ncols()))
    }
}

/// Slicing: `&mat[..b, ..d]`
impl<T, O> Index<(RangeTo<u32>, RangeTo<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeTo<u32>, RangeTo<u32>)) -> &::strided::Mat<T, O> {
        self.index((0..r.end, 0..c.end))
    }
}

/// Row slicing: `&mat[a..b, ..]`
impl<T, O> Index<(Range<u32>, RangeFull)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, _): (Range<u32>, RangeFull)) -> &::strided::Mat<T, O> {
        self.index(r)
    }
}

/// Row slicing: `&mat[a.., ..]`
impl<T, O> Index<(RangeFrom<u32>, RangeFull)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, _): (RangeFrom<u32>, RangeFull)) -> &::strided::Mat<T, O> {
        self.index(r)
    }
}

/// Row slicing: `&mat[..b, ..]`
impl<T, O> Index<(RangeTo<u32>, RangeFull)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, _): (RangeTo<u32>, RangeFull)) -> &::strided::Mat<T, O> {
        self.index(r)
    }
}

/// Column slicing: `&mat[.., a..b]`
impl<T, O> Index<(RangeFull, Range<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (_, c): (RangeFull, Range<u32>)) -> &::strided::Mat<T, O> {
        unsafe {
            &*self.col_slice_raw(c)
        }
    }
}

/// Column slicing: `&mat[.., a..]`
impl<T, O> Index<(RangeFull, RangeFrom<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeFull, RangeFrom<u32>)) -> &::strided::Mat<T, O> {
        self.index((r, c.start..self.ncols()))
    }
}

/// Column slicing: `&mat[.., ..b]`
impl<T, O> Index<(RangeFull, RangeTo<u32>)> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, (r, c): (RangeFull, RangeTo<u32>)) -> &::strided::Mat<T, O> {
        self.index((r, 0..c.end))
    }
}

/// Column indexing: `&mat[.., i]`
impl<T> Index<(RangeFull, u32)> for ::strided::Mat<T, ::order::Col> {
    type Output = ::Col<T>;

    fn index(&self, (_, i): (RangeFull, u32)) -> &::Col<T> {
        unsafe {
            &*self.col_raw(i)
        }
    }
}

/// Column indexing: `&mat[.., i]`
impl<T> Index<(RangeFull, u32)> for ::strided::Mat<T, ::order::Row> {
    type Output = ::strided::Col<T>;

    fn index(&self, (_, i): (RangeFull, u32)) -> &::strided::Col<T> {
        unsafe {
            &*self.strided_col_raw(i)
        }
    }
}

/// Row indexing: `&mat[i]`
impl<T> Index<u32> for ::strided::Mat<T, ::order::Col> {
    type Output = ::strided::Row<T>;

    fn index(&self, i: u32) -> &::strided::Row<T> {
        unsafe {
            &*self.strided_row_raw(i)
        }
    }
}

/// Row indexing: `&mat[i]`
impl<T> Index<u32> for ::strided::Mat<T, ::order::Row> {
    type Output = ::Row<T>;

    fn index(&self, i: u32) -> &::Row<T> {
        unsafe {
            &*self.row_raw(i)
        }
    }
}

/// Element indexing: `&mat[i, j]`
impl<T, O> Index<(u32, u32)> for ::strided::Mat<T, O> where O: Order {
    type Output = T;

    fn index(&self, (r, c): (u32, u32)) -> &T {
        unsafe {
            &*self.index_raw((r, c))
        }
    }
}

/// Row slicing: `&mat[a..b]`
impl<T, O> Index<Range<u32>> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, r: Range<u32>) -> &::strided::Mat<T, O> {
        unsafe {
            &*self.row_slice_raw(r)
        }
    }
}

/// Row slicing: `&mat[a..]`
impl<T, O> Index<RangeFrom<u32>> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, r: RangeFrom<u32>) -> &::strided::Mat<T, O> {
        &self[r.start..self.nrows()]
    }
}

/// Row slicing: `&mat[..b]`
impl<T, O> Index<RangeTo<u32>> for ::strided::Mat<T, O> where O: Order {
    type Output = ::strided::Mat<T, O>;

    fn index(&self, r: RangeTo<u32>) -> &::strided::Mat<T, O> {
        &self[0..r.end]
    }
}

/// Setting an element: `mat[i, j] = x`
impl<T, O> IndexAssign<(u32, u32), T> for ::strided::Mat<T, O> where O: Order {
    fn index_assign(&mut self, (i, j): (u32, u32), rhs: T) {
        *self.index_mut((i, j)) = rhs;
    }
}

/// Mutable slicing: `&mut mat[a..b, c..d]`
impl<T, O> IndexMut<(Range<u32>, Range<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (Range<u32>, Range<u32>)) -> &mut ::strided::Mat<T, O> {
        unsafe {
            &mut *self.slice_raw((r, c))
        }
    }
}

/// Mutable slicing: `&mut mat[a..b, c..]`
impl<T, O> IndexMut<(Range<u32>, RangeFrom<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (Range<u32>, RangeFrom<u32>)) -> &mut ::strided::Mat<T, O> {
        let end = self.ncols();
        self.index_mut((r, c.start..end))
    }
}

/// Mutable slicing: `&mut mat[a..b, ..d]`
impl<T, O> IndexMut<(Range<u32>, RangeTo<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (Range<u32>, RangeTo<u32>)) -> &mut ::strided::Mat<T, O> {
        self.index_mut((r, 0..c.end))
    }
}

/// Mutable slicing: `&mut mat[a.., c..d]`
impl<T, O> IndexMut<(RangeFrom<u32>, Range<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeFrom<u32>, Range<u32>)) -> &mut ::strided::Mat<T, O> {
        let end = self.nrows();
        self.index_mut((r.start..end, c))
    }
}

/// Mutable slicing: `&mut mat[a.., c..]`
impl<T, O> IndexMut<(RangeFrom<u32>, RangeFrom<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeFrom<u32>, RangeFrom<u32>)) -> &mut ::strided::Mat<T, O> {
        let (rend, cend) = self.size();
        self.index_mut((r.start..rend, c.start..cend))
    }
}

/// Mutable slicing: `&mut mat[a.., ..d]`
impl<T, O> IndexMut<(RangeFrom<u32>, RangeTo<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeFrom<u32>, RangeTo<u32>)) -> &mut ::strided::Mat<T, O> {
        let end = self.nrows();
        self.index_mut((r.start..end, 0..c.end))
    }
}

/// Mutable slicing: `&mut mat[..b, c..d]`
impl<T, O> IndexMut<(RangeTo<u32>, Range<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeTo<u32>, Range<u32>)) -> &mut ::strided::Mat<T, O> {
        self.index_mut((0..r.end, c))
    }
}

/// Mutable slicing: `&mut mat[..b, c..]`
impl<T, O> IndexMut<(RangeTo<u32>, RangeFrom<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeTo<u32>, RangeFrom<u32>)) -> &mut ::strided::Mat<T, O> {
        let end = self.ncols();
        self.index_mut((0..r.end, c.start..end))
    }
}

/// Mutable slicing: `&mut mat[..b, ..d]`
impl<T, O> IndexMut<(RangeTo<u32>, RangeTo<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeTo<u32>, RangeTo<u32>)) -> &mut ::strided::Mat<T, O> {
        self.index_mut((0..r.end, 0..c.end))
    }
}

/// Mutable row slicing: `&mut mat[a..b, ..]`
impl<T, O> IndexMut<(Range<u32>, RangeFull)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, _): (Range<u32>, RangeFull)) -> &mut ::strided::Mat<T, O> {
        self.index_mut(r)
    }
}

/// Mutable row slicing: `&mut mat[a.., ..]`
impl<T, O> IndexMut<(RangeFrom<u32>, RangeFull)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, _): (RangeFrom<u32>, RangeFull)) -> &mut ::strided::Mat<T, O> {
        self.index_mut(r)
    }
}

/// Mutable row slicing: `&mut mat[..b, ..]`
impl<T, O> IndexMut<(RangeTo<u32>, RangeFull)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, _): (RangeTo<u32>, RangeFull)) -> &mut ::strided::Mat<T, O> {
        self.index_mut(r)
    }
}

/// Mutable column slicing: `&mut mat[.., a..b]`
impl<T, O> IndexMut<(RangeFull, Range<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (_, c): (RangeFull, Range<u32>)) -> &mut ::strided::Mat<T, O> {
        unsafe {
            &mut *self.col_slice_raw(c)
        }
    }
}

/// Mutable column slicing: `&mut mat[.., a..]`
impl<T, O> IndexMut<(RangeFull, RangeFrom<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeFull, RangeFrom<u32>)) -> &mut ::strided::Mat<T, O> {
        let end = self.ncols();
        self.index_mut((r, c.start..end))
    }
}

/// Mutable column slicing: `&mut mat[.., ..b]`
impl<T, O> IndexMut<(RangeFull, RangeTo<u32>)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (RangeFull, RangeTo<u32>)) -> &mut ::strided::Mat<T, O> {
        self.index_mut((r, 0..c.end))
    }
}

/// Mutable column indexing: `&mut mat[.., i]`
impl<T> IndexMut<(RangeFull, u32)> for ::strided::Mat<T, ::order::Col> {
    fn index_mut(&mut self, (_, i): (RangeFull, u32)) -> &mut ::Col<T> {
        unsafe {
            &mut *self.col_raw(i)
        }
    }
}

/// Mutable column indexing: `&mut mat[.., i]`
impl<T> IndexMut<(RangeFull, u32)> for ::strided::Mat<T, ::order::Row> {
    fn index_mut(&mut self, (_, i): (RangeFull, u32)) -> &mut ::strided::Col<T> {
        unsafe {
            &mut *self.strided_col_raw(i)
        }
    }
}

/// Mutable row indexing: `&mut mat[i]`
impl<T> IndexMut<u32> for ::strided::Mat<T, ::order::Col> {
    fn index_mut(&mut self, i: u32) -> &mut ::strided::Row<T> {
        unsafe {
            &mut *self.strided_row_raw(i)
        }
    }
}

/// Mutable row indexing: `&mut mat[i]`
impl<T> IndexMut<u32> for ::strided::Mat<T, ::order::Row> {
    fn index_mut(&mut self, i: u32) -> &mut ::Row<T> {
        unsafe {
            &mut *self.row_raw(i)
        }
    }
}

/// Mutable element indexing: `&mut mat[i, j]`
impl<T, O> IndexMut<(u32, u32)> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, (r, c): (u32, u32)) -> &mut T {
        unsafe {
            &mut *self.index_raw((r, c))
        }
    }
}

/// Mutable row slicing: `&mut mat[a..b]`
impl<T, O> IndexMut<Range<u32>> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, r: Range<u32>) -> &mut ::strided::Mat<T, O> {
        unsafe {
            &mut *self.row_slice_raw(r)
        }
    }
}

/// Mutable row slicing: `&mut mat[a..]`
impl<T, O> IndexMut<RangeFrom<u32>> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, r: RangeFrom<u32>) -> &mut ::strided::Mat<T, O> {
        let end = self.nrows();
        &mut self[r.start..end]
    }
}

/// Mutable row slicing: `&mut mat[..b]`
impl<T, O> IndexMut<RangeTo<u32>> for ::strided::Mat<T, O> where O: Order {
    fn index_mut(&mut self, r: RangeTo<u32>) -> &mut ::strided::Mat<T, O> {
        &mut self[0..r.end]
    }
}

impl<'a, T, O> IntoCow<'a, ::strided::Mat<T, O>> for &'a ::strided::Mat<T, O> where
    T: Clone, O: Order,
{
    fn into_cow(self) -> Cow<'a, ::strided::Mat<T, O>> {
        Cow::Borrowed(self)
    }
}

impl<'a, T, O> IntoCow<'static, ::strided::Mat<T, O>> for Box<::Mat<T, O>> where
    T: Clone, O: Order,
{
    fn into_cow(self) -> Cow<'static, ::strided::Mat<T, O>> {
        Cow::Owned(self)
    }
}

impl<'a, T, O> IntoIterator for &'a ::strided::Mat<T, O> where O: Order {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, O>;

    fn into_iter(self) -> Iter<'a, T, O> {
        self.iter()
    }
}

impl<'a, T, O> IntoIterator for &'a mut ::strided::Mat<T, O> where O: Order {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, O>;

    fn into_iter(self) -> IterMut<'a, T, O> {
        self.iter_mut()
    }
}

impl<T, O> Matrix for ::strided::Mat<T, O> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        self.repr().info.nrows.u32()
    }

    fn ncols(&self) -> u32 {
        self.repr().info.ncols.u32()
    }
}

unsafe impl<T: Send, O> Send for ::strided::Mat<T, O> {}

unsafe impl<T: Sync, O> Sync for ::strided::Mat<T, O> {}

impl<T, O> ToOwned for ::strided::Mat<T, O> where T: Clone, O: Order {
    type Owned = Box<::Mat<T, O>>;

    fn to_owned(&self) -> Box<::Mat<T, O>> {
        let FatPtr { info, .. } = self.repr();

        if let Some(slice) = self.as_slice() {
            let mut v = slice.to_owned();
            let data = v.as_mut_ptr();
            mem::forget(v);

            unsafe {
                Box::from_raw(fat_ptr::new(FatPtr {
                    data: data,
                    info: ::mat::Info {
                        _marker: info._marker,
                        ncols: info.ncols,
                        nrows: info.nrows,
                    },
                }))
            }
        } else {
            unimplemented!()
        }
    }
}

impl<'a, T, O> Transpose for &'a ::strided::Mat<T, O> where O: Order {
    type Output = &'a ::strided::Mat<T, O::Transposed>;

    fn t(self) -> &'a ::strided::Mat<T, O::Transposed> {
        unsafe {
            &*self.t_raw()
        }
    }
}

impl<'a, T, O> Transpose for &'a mut ::strided::Mat<T, O> where O: Order {
    type Output = &'a mut ::strided::Mat<T, O::Transposed>;

    fn t(self) -> &'a mut ::strided::Mat<T, O::Transposed> {
        unsafe {
            &mut *self.t_raw()
        }
    }
}

impl<T, O> Unsized for ::strided::Mat<T, O> {
    type Data = T;
    type Info = Info<O>;

    fn size_of_val(_: Info<O>) -> usize {
        // NB unimplemented for now to avoid cascading the `Order` bound everywhere
        unimplemented!()
        //mem::size_of::<T>() * info.stride.usize() * match O::order() {
            //::Order::Col => info.ncols.usize(),
            //::Order::Row => info.nrows.usize(),
        //}
    }
}
