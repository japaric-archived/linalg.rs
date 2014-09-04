use notsafe::{UnsafeIndex, UnsafeIndexMut};
use traits::{Matrix, OptionIndex, OptionIndexMut};
use {Mat, MutView, View};

// XXX I can't of a way to implement this with traits, so I'll repeat myself using macros

macro_rules! index {
    () => {
        fn index(&self, &(row, col): &(uint, uint)) -> &T {
            let (nrows, ncols) = self.size();

            assert!(row < nrows && col < ncols);

            unsafe { self.unsafe_index(&(row, col)) }
        }
    }
}

macro_rules! at {
    () => {
        fn at(&self, &(row, col): &(uint, uint)) -> Option<&T> {
            let (nrows, ncols) = self.size();

            if row < nrows && col < ncols {
                Some(unsafe { self.unsafe_index(&(row, col)) })
            } else {
                None
            }
        }
    }
}

impl<'a, T> Index<(uint, uint), T> for MutView<'a, T> { index!() }
impl<'a, T> Index<(uint, uint), T> for View<'a, T> { index!() }
impl<'a, T> OptionIndex<(uint, uint), T> for MutView<'a, T> { at!() }
impl<'a, T> OptionIndex<(uint, uint), T> for View<'a, T> { at!() }
impl<T> Index<(uint, uint), T> for Mat<T> { index!() }
impl<T> OptionIndex<(uint, uint), T> for Mat<T> { at!() }

macro_rules! index_mut {
    () => {
        fn index_mut(&mut self, &(row, col): &(uint, uint)) -> &mut T {
            let (nrows, ncols) = self.size();

            assert!(row < nrows && col < ncols);

            unsafe { self.unsafe_index_mut(&(row, col)) }
        }
    }
}

macro_rules! at_mut {
    () => {
        fn at_mut(&mut self, &(row, col): &(uint, uint)) -> Option<&mut T> {
            let (nrows, ncols) = self.size();

            if row < nrows && col < ncols {
                Some(unsafe { self.unsafe_index_mut(&(row, col)) })
            } else {
                None
            }
        }
    }
}

impl<'a, T> IndexMut<(uint, uint), T> for MutView<'a, T> { index_mut!() }
impl<'a, T> OptionIndexMut<(uint, uint), T> for MutView<'a, T> { at_mut!() }
impl<T> IndexMut<(uint, uint), T> for Mat<T> { index_mut!() }
impl<T> OptionIndexMut<(uint, uint), T> for Mat<T> { at_mut!() }
