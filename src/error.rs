//! Single errors as unit structs

use Error;

#[derive(Copy, Debug)]
pub struct OutOfBounds;

impl From<OutOfBounds> for Error {
    fn from(_: OutOfBounds) -> Error {
        Error::OutOfBounds
    }
}
