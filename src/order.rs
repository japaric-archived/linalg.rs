//! Order used to lay out a matrix in memory

// FIXME This should be a "sealed" trait
/// Memory order
pub trait Order {
    /// New order after transposing
    type Transposed;

    #[doc(hidden)]
    fn order() -> ::Order;
}

/// Column major (a.k.a. "Fortran") order (default)
#[derive(Debug)]
pub enum Col {}

impl Order for Col {
    type Transposed = Row;

    #[doc(hidden)]
    fn order() -> ::Order {
        ::Order::Col
    }
}

/// Row major (a.k.a. "C") order
#[derive(Debug)]
pub enum Row {}

impl Order for Row {
    type Transposed = Col;

    #[doc(hidden)]
    fn order() -> ::Order {
        ::Order::Row
    }
}
