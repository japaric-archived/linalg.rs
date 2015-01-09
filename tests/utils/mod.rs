pub trait IsWithin {
    fn is_within(&self, other: Self) -> bool;
}

impl IsWithin for (usize, usize) {
    fn is_within(&self, other: (usize, usize)) -> bool {
        self.0 <= other.0 && self.1 <= other.1
    }
}
