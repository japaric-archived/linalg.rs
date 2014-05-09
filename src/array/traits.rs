pub trait ArrayDot<T, U> {
    fn dot(&self, rhs: &T) -> U;
}

pub trait ArrayNorm2<T> {
    fn norm2(&self) -> T;
}

pub trait ArrayScale<T> {
    fn scale(&mut self, alpha: T);
}

pub trait ArrayShape<S> {
    fn shape(&self) -> S;
}
