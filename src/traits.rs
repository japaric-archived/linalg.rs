// FIXME mozilla/rust#5992 Use std AddAssign
pub trait AddAssign<R> {
    fn add_assign(&mut self, other: &R);
}

// FIXME mozilla/rust#6515 Use std Index
pub trait Index<I, T> {
    fn index<'a>(&'a self, index: &I) -> &'a T;
}

pub trait UnsafeIndex<I, T> {
    unsafe fn unsafe_index<'a>(&'a self, index: &I) -> &'a T;
}

pub trait Iterable<'a, T, I: Iterator<&'a T>> {
    fn iter(&'a self) -> I;

    #[inline]
    fn all(&'a self, pred: |&'a T| -> bool) -> bool {
        self.iter().all(pred)
    }

    #[inline]
    fn any(&'a self, pred: |&'a T| -> bool) -> bool {
        self.iter().any(pred)
    }
}

// FIXME mozilla/rust#5992 Use std MulAssign
pub trait MulAssign<R> {
    fn mul_assign(&mut self, other: &R);
}

// FIXME mozilla/rust#5992 Use std SubAssign
pub trait SubAssign<R> {
    fn sub_assign(&mut self, other: &R);
}
