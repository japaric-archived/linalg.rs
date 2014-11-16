#![feature(globs, macro_rules, phase, tuple_indexing)]

extern crate linalg;
extern crate onezero;
extern crate quickcheck;
#[phase(plugin)]
extern crate quickcheck_macros;

mod setup;

mod view_mul;
