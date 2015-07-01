//! "Strided" data structures

mod col;
mod diag;
mod row;
mod stripes;

pub mod mat;
pub mod vector;

/// Strided column vector
#[derive(Debug)]
pub struct Col<T>(Vector<T>);

/// Matrix diagonal
#[derive(Debug)]
pub struct Diag<T>(Vector<T>);

/// Iterator over a matrix in horizontal (non-overlapping) stripes
pub struct HStripes<'a, T: 'a, O: 'a> {
    m: &'a ::strided::Mat<T, O>,
    size: u32,
}

/// Iterator over a matrix in horizontal (non-overlapping) mutable stripes
pub struct HStripesMut<'a, T: 'a, O: 'a> {
    m: &'a mut ::strided::Mat<T, O>,
    size: u32,
}

/// Strided matrix
pub unsized type Mat<T, O>;

/// Strided row vector
#[derive(Debug)]
pub struct Row<T>(Vector<T>);

/// Either a diagonal, a row vector or a column vector
pub unsized type Vector<T>;

/// Iterator over a matrix in vertical (non-overlapping) stripes
pub struct VStripes<'a, T: 'a, O: 'a> {
    m: &'a ::strided::Mat<T, O>,
    size: u32,
}

/// Iterator over a matrix in vertical (non-overlapping) mutable stripes
pub struct VStripesMut<'a, T: 'a, O: 'a> {
    m: &'a mut ::strided::Mat<T, O>,
    size: u32,
}
