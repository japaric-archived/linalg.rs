//! Test `mat!` error messages

#![feature(plugin)]
#![plugin(linalg_macros)]

extern crate linalg;

fn short_row() -> Mat<i32> {
    mat![  //~ error: expected 3 columns, but found 2
        1, 2, 3;
        4, 5;
    ]
}

fn empty() -> Mat<i32> {
    mat![]  //~ error: empty matrix
}

fn missing_element() -> Mat<i32> {
    mat![  //~ error: no element found at (0, 1)
        1,  , 3;
        4, 5, 6;
    ]
}

fn trailing_comma() -> RowVec<i32> {
    mat![1, 2, 3,]  //~ error: trailing comma
}

fn main() {}
