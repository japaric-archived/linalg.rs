//! Single errors as unit structs

use Error;

#[derive(Clone, Copy, Debug)]
pub struct OutOfBounds;

impl From<OutOfBounds> for Error {
    fn from(_: OutOfBounds) -> Error {
        Error::OutOfBounds
    }
}
