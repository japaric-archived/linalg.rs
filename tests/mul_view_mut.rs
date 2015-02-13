#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate linalg;
extern crate onezero;
extern crate quickcheck;
extern crate rand;

#[macro_use]
mod setup;

mod mutview_mul;
