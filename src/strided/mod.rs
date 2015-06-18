//! "Strided" data structures

mod col;
mod diag;
mod row;
mod stripes;

pub mod mat;
pub mod raw;
pub mod slice;

/// Strided column vector
// FIXME compiler bug
//pub struct Col<T>(Slice<T>);
pub unsized type Col<T> = raw::Slice<T>;

/// Matrix diagonal
//pub struct Diag<T>(Slice<T>);
pub unsized type Diag<T> = raw::Slice<T>;

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
pub unsized type Mat<T, O> = raw::Mat<T, O>;

/// Strided row vector
//pub struct Row<T>(Slice<T>);
pub unsized type Row<T> = raw::Slice<T>;

pub unsized type Slice<T> = raw::Slice<T>;

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
