use std::iter::order;
use std::kinds::marker;
use std::{cmp, mem};

use {Col, Diag, Error, Result, Row};
use error::OutOfBounds;

pub struct View<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    pub data: *mut T,
    pub ld: uint,  // Leading dimension
    pub ncols: uint,
    pub nrows: uint,
}

impl<'a, T> View<'a, T> {
    pub fn at(&self, (row, col): (uint, uint)) -> ::std::result::Result<&T, OutOfBounds> {
        let (nrows, ncols) = (self.nrows, self.ncols);

        if row < nrows && col < ncols {
            Ok(unsafe {
                mem::transmute(self.data.offset((col * self.ld + row) as int))
            })
        } else {
            Err(OutOfBounds)
        }
    }

    pub fn diag(&self, diag: int) -> Result<Diag<T>> {
        let (nrows, ncols) = (self.nrows, self.ncols);
        let stride = self.ld;

        if diag > 0 {
            let diag = diag as uint;

            if diag < ncols {
                Ok(Diag(unsafe { ::From::parts((
                    self.data.offset((diag * stride) as int) as *const _,
                    cmp::min(nrows, ncols - diag),
                    stride + 1,
                )) }))
            } else {
                Err(Error::NoSuchDiagonal)
            }
        } else {
            let diag = -diag as uint;

            if diag < nrows {
                Ok(Diag(unsafe { ::From::parts((
                    self.data.offset(diag as int) as *const _,
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
            _contravariant: marker::ContravariantLifetime,
            _nosend: marker::NoSend,
            col: 0,
            data: self.data,
            ld: self.ld,
            ncols: if self.nrows == 0 { 0 } else { self.ncols },
            nrows: self.nrows,
            row: 0,
        }
    }

    pub fn ncols(&self) -> uint {
        self.ncols
    }

    pub fn nrows(&self) -> uint {
        self.nrows
    }

    pub fn size(&self) -> (uint, uint) {
        (self.nrows, self.ncols)
    }

    pub fn slice(&self, start: (uint, uint), end: (uint, uint)) -> Result<View<T>> {
        let (end_row, end_col) = end;
        let (nrows, ncols) = (self.nrows, self.ncols);
        let (start_row, start_col) = start;

        if end_col > ncols || end_row > nrows {
            Err(Error::OutOfBounds)
        } else if start_col > end_col || start_row > end_row {
            Err(Error::InvalidSlice)
        } else {
            Ok(unsafe { ::From::parts((
                self.data.offset((start_col * self.ld + start_row) as int) as *const _,
                end_row - start_row,
                end_col - start_col,
                self.ld,
            ))})
        }
    }

    pub unsafe fn unsafe_col(&self, col: uint) -> Col<T> {
        Col(::From::parts((
            self.data.offset((col * self.ld) as int) as *const T,
            self.nrows(),
            1,
        )))
    }

    pub unsafe fn unsafe_row(&self, row: uint) -> Row<T> {
        Row(::From::parts((
            self.data.offset(row as int) as *const _,
            self.ncols(),
            self.ld,
        )))
    }
}

impl<'a, T> Copy for View<'a, T> {}

impl<'a, T> ::From<(*const T, uint, uint, uint)> for View<'a, T> {
    unsafe fn parts((data, nrows, ncols, ld): (*const T, uint, uint, uint)) -> View<'a, T> {
        View {
            _contravariant: marker::ContravariantLifetime,
            _nosend: marker::NoSend,
            data: data as *mut _,
            ld: ld,
            ncols: ncols,
            nrows: nrows,
        }
    }
}

impl<'a, T> PartialEq for View<'a, T> where T: PartialEq {
    fn eq(&self, rhs: &View<'a, T>) -> bool {
        self.size() == rhs.size() && order::eq(self.iter(), rhs.iter())
    }
}

pub struct Items<'a, T: 'a> {
    _contravariant: marker::ContravariantLifetime<'a>,
    _nosend: marker::NoSend,
    col: uint,
    data: *const T,
    ld: uint,
    ncols: uint,
    nrows: uint,
    row: uint,
}

impl<'a, T> Copy for Items<'a, T> {}

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
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
                mem::transmute(self.data.offset((col * self.ld + row) as int))
            })
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let total = self.nrows * self.ncols;
        let done = self.row * self.ncols + self.col;
        let left = total - done;

        (left, Some(left))
    }
}
