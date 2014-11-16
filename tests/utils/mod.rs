pub trait IsWithin {
    fn is_within(&self, other: Self) -> bool;
}

impl IsWithin for (uint, uint) {
    fn is_within(&self, other: (uint, uint)) -> bool {
        self.0 <= other.0 && self.1 <= other.1
    }
}
