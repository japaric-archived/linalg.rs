use std::iter::order;
use std::marker;
use std::ops::Range;
use std::{cmp, mem};

use {Col, Diag, Error, Result, Row};
use error::OutOfBounds;

pub struct View<'a, T> {
    _marker: marker::PhantomData<fn() -> &'a T>,
    pub data: *mut T,
    pub ld: usize,  // Leading dimension
    pub ncols: usize,
    pub nrows: usize,
}

impl<'a, T> View<'a, T> {
    pub fn at(&self, (row, col): (usize, usize)) -> ::std::result::Result<&T, OutOfBounds> {
        let (nrows, ncols) = (self.nrows, self.ncols);

        if row < nrows && col < ncols {
            Ok(unsafe {
                mem::transmute(self.data.offset((col * self.ld + row) as isize))
            })
        } else {
            Err(OutOfBounds)
        }
    }

    pub fn diag(&self, diag: isize) -> Result<Diag<T>> {
        let (nrows, ncols) = (self.nrows, self.ncols);
        let stride = self.ld;

        if diag > 0 {
            let diag = diag as usize;

            if diag < ncols {
                Ok(Diag(unsafe { ::From::parts((
                    self.data.offset((diag * stride) as isize) as *const _,
                    cmp::min(nrows, ncols - diag),
                    stride + 1,
                )) }))
            } else {
                Err(Error::NoSuchDiagonal)
            }
        } else {
            let diag = -diag as usize;

            if diag < nrows {
                Ok(Diag(unsafe { ::From::parts((
                    self.data.offset(diag as isize) as *const _,
                    cmp::min(nrows - diag, ncols),
                    stride + 1,
                )) }))
            } else {
                Err(Error::NoSuchDiagonal)
            }
        }
    }

    pub fn iter(&self) -> Items<T> {
        Items {
            _marker: marker::PhantomData,
            col: 0,
            data: self.data,
            ld: self.ld,
            ncols: if self.nrows == 0 { 0 } else { self.ncols },
            nrows: self.nrows,
            row: 0,
        }
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn size(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    pub fn slice(&self, Range { start, end }: Range<(usize, usize)>) -> Result<View<T>> {
        let (end_row, end_col) = end;
        let (nrows, ncols) = (self.nrows, self.ncols);
        let (start_row, start_col) = start;

        if end_col > ncols || end_row > nrows {
            Err(Error::OutOfBounds)
        } else if start_col > end_col || start_row > end_row {
            Err(Error::InvalidSlice)
        } else {
            Ok(unsafe { ::From::parts((
                self.data.offset((start_col * self.ld + start_row) as isize) as *const _,
                end_row - start_row,
                end_col - start_col,
                self.ld,
            ))})
        }
    }

    pub unsafe fn unsafe_col(&self, col: usize) -> Col<T> {
        Col(::From::parts((
            self.data.offset((col * self.ld) as isize) as *const T,
            self.nrows(),
            1,
        )))
    }

    pub unsafe fn unsafe_row(&self, row: usize) -> Row<T> {
        Row(::From::parts((
            self.data.offset(row as isize) as *const _,
            self.ncols(),
            self.ld,
        )))
    }
}

impl<'a, T> Copy for View<'a, T> {}

impl<'a, T> ::From<(*const T, usize, usize, usize)> for View<'a, T> {
    unsafe fn parts((data, nrows, ncols, ld): (*const T, usize, usize, usize)) -> View<'a, T> {
        View {
            _marker: marker::PhantomData,
            data: data as *mut _,
            ld: ld,
            ncols: ncols,
            nrows: nrows,
        }
    }
}

impl<'a, 'b, T, U> PartialEq<View<'a, T>> for View<'b, U> where U: PartialEq<T> {
    fn eq(&self, rhs: &View<T>) -> bool {
        self.size() == rhs.size() && order::eq(self.iter(), rhs.iter())
    }
}

pub struct Items<'a, T> {
    _marker: marker::PhantomData<fn() -> &'a T>,
    col: usize,
    data: *const T,
    ld: usize,
    ncols: usize,
    nrows: usize,
    row: usize,
}

impl<'a, T> Copy for Items<'a, T> {}

impl<'a, T> Iterator for Items<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.col == self.ncols {
            None
        } else {
            let (row, col) = (self.row, self.col);

            self.row += 1;

            if self.row == self.nrows {
                self.row = 0;
                self.col += 1;
            }

            Some(unsafe {
                mem::transmute(self.data.offset((col * self.ld + row) as isize))
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = self.nrows * self.ncols;
        let done = self.row * self.ncols + self.col;
        let left = total - done;

        (left, Some(left))
    }
}
