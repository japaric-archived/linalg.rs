pub trait ArrayNorm2<T> {
    fn norm2(&self) -> T;
}

pub trait ArrayScale<T> {
    fn scale(&mut self, alpha: T);
}
