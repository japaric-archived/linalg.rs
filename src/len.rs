use traits::Collection;
use {Col, Diag, Row};

impl<'a, T> Collection for &'a [T] {
    fn len(&self) -> uint {
        SliceExt::len(*self)
    }
}

impl<'a, T> Collection for &'a mut [T] {
    fn len(&self) -> uint {
        SliceExt::len(*self)
    }
}

impl<T> Collection for Box<[T]> {
    fn len(&self) -> uint {
        SliceExt::len(&**self)
    }
}

macro_rules! impls {
    ($($ty:ty),+) => {$(
        impl<V> Collection for $ty where V: Collection {
            fn len(&self) -> uint {
                self.0.len()
            }
        })+
    }
}

impls!(Col<V>, Diag<V>, Row<V>);
