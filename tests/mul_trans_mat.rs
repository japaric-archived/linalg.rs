#![allow(unstable)]
#![feature(plugin)]

extern crate linalg;
extern crate onezero;
extern crate quickcheck;
#[plugin]
extern crate quickcheck_macros;

#[macro_use]
mod setup;

mod trans_mat_mul;
