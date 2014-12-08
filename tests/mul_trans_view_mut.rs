#![feature(globs, macro_rules, phase)]

extern crate linalg;
extern crate onezero;
extern crate quickcheck;
#[phase(plugin)]
extern crate quickcheck_macros;

mod setup;

mod trans_mutview_mul;
