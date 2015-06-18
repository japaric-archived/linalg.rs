use std::ops::{IndexAssign, Range, RangeFrom, RangeFull, RangeTo};

use extract::Extract;

// Single contiguous iterator pass, should optimize to memcpy
// NOTE Core
impl<'a, T> IndexAssign<RangeFull, &'a ::Row<T>> for ::Row<T> where T: Clone {
    fn index_assign(&mut self, _: RangeFull, rhs: &::Row<T>) {
        unsafe {
            assert_eq!(self.len(), rhs.len());

            let mut src = rhs.iter();
            for dst in self {
                *dst = src.next().extract().clone();
            }
        }
    }
}

// NOTE Secondary
impl<'a, T> IndexAssign<Range<u32>, &'a ::Row<T>> for ::Row<T> where T: Clone {
    fn index_assign(&mut self, r: Range<u32>, rhs: &::Row<T>) {
        ::Row::index_assign(&mut self[r], .., rhs)
    }
}

// NOTE Secondary
impl<'a, T> IndexAssign<RangeFrom<u32>, &'a ::Row<T>> for ::Row<T> where T: Clone {
    fn index_assign(&mut self, r: RangeFrom<u32>, rhs: &::Row<T>) {
        ::Row::index_assign(&mut self[r], .., rhs)
    }
}

// NOTE Secondary
impl<'a, T> IndexAssign<RangeTo<u32>, &'a ::Row<T>> for ::Row<T> where T: Clone {
    fn index_assign(&mut self, r: RangeTo<u32>, rhs: &::Row<T>) {
        ::Row::index_assign(&mut self[r], .., rhs)
    }
}

// Single contiguous iterator pass, should optimize to memset
// NOTE Core
impl<T> IndexAssign<RangeFull, T> for ::Row<T> where T: Clone {
    fn index_assign(&mut self, _: RangeFull, rhs: T) {
        let ref src = rhs;

        for dst in self {
            *dst = src.clone();
        }
    }
}
