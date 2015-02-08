#![feature(plugin)]

extern crate linalg;
extern crate onezero;
extern crate quickcheck;
#[plugin]
extern crate quickcheck_macros;
extern crate rand;

#[macro_use]
mod setup;

mod trans_view_mul;
