extern crate linalg;

use linalg::prelude::*;

// Test that none of the reference-like structs can outlive their referenced data

fn col(c: ColVec<f32>) {
    let col = {
        let c = c.clone();

        c.slice(0..2)  //~ error: does not live long enough
    };

    let col_mut = {
        let c = c.clone();

        c.slice_mut(0..2)  //~ error: does not live long enough
    };
}

fn mat(m: Mat<f32>) {
    let col = {
        let m = m.clone();

        m.col(0)  //~ error: does not live long enough
    };

    let col_mut = {
        let mut m = m.clone();

        m.col_mut(0)  //~ error: does not live long enough
    };

    let diag = {
        let m = m.clone();

        m.diag(0)  //~ error: does not live long enough
    };

    let diag_mut = {
        let mut m = m.clone();

        m.diag_mut(0)  //~ error: does not live long enough
    };

    let row = {
        let m = m.clone();

        m.row(0)  //~ error: does not live long enough
    };

    let row_mut = {
        let mut m = m.clone();

        m.row_mut(0)  //~ error: does not live long enough
    };

    let view = {
        let m = m.clone();

        m.slice((0, 0)..(2, 2))  //~ error: does not live long enough
    };

    let view_mut = {
        let mut m = m.clone();

        m.slice_mut((0, 0)..(2, 2))  //~ error: does not live long enough
    };
}

fn row(r: RowVec<f32>) {
    let row = {
        let r = r.clone();

        r.slice(0..2)  //~ error: does not live long enough
    };

    let row_mut = {
        let r = r.clone();

        r.slice_mut(0..2)  //~ error: does not live long enough
    };
}

fn main() {}
