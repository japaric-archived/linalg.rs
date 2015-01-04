//! BLAS acceleration

pub mod axpy;
pub mod copy;
pub mod dot;
pub mod gemm;
pub mod gemv;

mod ffi;

#[repr(i8)]
/// Transpose matrix before operation?
#[derive(Copy)]
pub enum Transpose {
    /// Don't transpose
    No = 110, // 'n'
    /// Transpose
    Yes = 116,  // 't'
}

/// The integer used by the BLAS library
// TODO Handle 64-bit BLAS
#[allow(non_camel_case_types)]
pub type blasint = ::libc::c_int;
