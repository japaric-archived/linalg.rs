#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate approx;
extern crate linalg;
extern crate onezero;
extern crate quickcheck;
extern crate rand;

#[macro_use]
mod setup;

mod trans_mutview_mul;
