//! Single errors as unit structs

use std::error::FromError;

use Error;

#[derive(Copy, Show)]
pub struct OutOfBounds;

impl FromError<OutOfBounds> for Error {
    fn from_error(_: OutOfBounds) -> Error {
        Error::OutOfBounds
    }
}
