use std::{mem, raw};

use {Col, Row};

pub trait UnsafeIndex<I, R> {
    unsafe fn unsafe_index<'a>(&'a self, index: &I) -> &'a R;
}

macro_rules! unsafe_index {
    () => {
        unsafe fn unsafe_index(&self, &index: &uint) -> &T {
            mem::transmute(self.as_ptr().offset(index as int))
        }
    }
}

impl<'a, T> UnsafeIndex<uint, T> for &'a [T] { unsafe_index!() }
impl<'a, T> UnsafeIndex<uint, T> for &'a mut [T] { unsafe_index!() }
impl<T> UnsafeIndex<uint, T> for Vec<T> { unsafe_index!() }

pub trait UnsafeIndexMut<I, R> {
    unsafe fn unsafe_index_mut<'a>(&'a mut self, index: &I) -> &'a mut R;
}

macro_rules! unsafe_index_mut {
    () => {
        unsafe fn unsafe_index_mut(&mut self, &index: &uint) -> &mut T {
            mem::transmute(self.as_mut_ptr().offset(index as int))
        }
    }
}

impl<'a, T> UnsafeIndexMut<uint, T> for &'a mut [T] { unsafe_index_mut!() }
impl<T> UnsafeIndexMut<uint, T> for Vec<T> { unsafe_index_mut!() }

pub trait UnsafeMatrixCol<'a, D> {
    unsafe fn unsafe_col(&'a self, col: uint) -> Col<D>;
}

pub trait UnsafeMatrixMutCol<'a, D> {
    unsafe fn unsafe_mut_col(&'a mut self, col: uint) -> Col<D>;
}

pub trait UnsafeMatrixMutRow<'a, D> {
    unsafe fn unsafe_mut_row(&'a mut self, row: uint) -> Row<D>;
}

pub trait UnsafeMutSlice<'a, I, S> {
    unsafe fn unsafe_mut_slice(&'a mut self, start: I, end: I) -> S;
}

macro_rules! unsafe_mut_slice {
    () => {
        unsafe fn unsafe_mut_slice(&mut self, start: uint, end: uint) -> &mut [T] {
            mem::transmute(raw::Slice {
                data: self.as_ptr().offset(start as int),
                len: end - start,
            })
        }
    }
}

impl<'a, 'b, T> UnsafeMutSlice<'b, uint, &'b mut [T]> for &'a mut [T] { unsafe_mut_slice!() }
impl<'a, T> UnsafeMutSlice<'a, uint, &'a mut [T]> for Vec<T>  { unsafe_mut_slice!() }

pub trait UnsafeMatrixRow<'a, D> {
    unsafe fn unsafe_row(&'a self, row: uint) -> Row<D>;
}

pub trait UnsafeSlice<'a, I, S> {
    unsafe fn unsafe_slice(&'a self, start: I, end: I) -> S;
}

macro_rules! unsafe_slice {
    () => {
        unsafe fn unsafe_slice(&self, start: uint, end: uint) -> &[T] {
            mem::transmute(raw::Slice {
                data: self.as_ptr().offset(start as int),
                len: end - start,
            })
        }
    }
}

impl<'a, 'b, T> UnsafeSlice<'b, uint, &'b [T]> for &'a [T] { unsafe_slice!() }
impl<'a, 'b, T> UnsafeSlice<'b, uint, &'b [T]> for &'a mut [T] { unsafe_slice!() }
impl<'a, T> UnsafeSlice<'a, uint, &'a [T]> for Vec<T> { unsafe_slice!() }
