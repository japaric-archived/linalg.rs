//! Test that the reference-like structs are `Send`

#![allow(dead_code)]

extern crate linalg;

use linalg::{Col, ColMut, Row, RowMut, SubMat, SubMatMut};

fn is_send<T>(_: T) where T: Send {}

fn col<T>(c: Col<T>) where T: Sync {
    is_send(c);
}

fn col_mut<T>(c: ColMut<T>) where T: Send {
    is_send(c);
}

fn row<T>(r: Row<T>) where T: Sync {
    is_send(r);
}

fn row_mut<T>(r: RowMut<T>) where T: Send {
    is_send(r);
}

fn submat<T>(r: SubMat<T>) where T: Sync {
    is_send(r);
}

fn submat_mut<T>(r: SubMatMut<T>) where T: Send {
    is_send(r);
}
