#![feature(globs, macro_rules)]

extern crate linalg;

use linalg::prelude::*;

macro_rules! eq {
    ($lhs:expr, $rhs:expr,) => { assert_eq!($lhs.to_string(), $rhs.to_string()) }
}

mod col {
    use linalg::prelude::*;

    // Test that `Show` is correct for `ColVec`
    #[test]
    fn owned() {
        eq! {
            ColVec::from_fn(3, |i| i),
            "Col([0, 1, 2])",
        }
    }

    // Test that `Show` is correct for `Col`
    #[test]
    fn slice() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().col(0).unwrap(),
            "Col([(0, 0), (1, 0), (2, 0)])",
        }
    }

    // Test that `Show` is correct for `MutCol`
    #[test]
    fn slice_mut() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().col_mut(0).unwrap(),
            "Col([(0, 0), (1, 0), (2, 0)])",
        }
    }

    // Test that `Show` is correct for `strided::Col`
    #[test]
    fn strided() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().t().col(0).unwrap(),
            "Col([(0, 0), (0, 1), (0, 2)])",
        }
    }

    // Test that `Show` is correct for `strided::MutCol`
    #[test]
    fn strided_mut() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().t().col_mut(0).unwrap(),
            "Col([(0, 0), (0, 1), (0, 2)])",
        }
    }
}

mod diag {
    use linalg::prelude::*;

    // Test that `Show` is correct for `Diag`
    #[test]
    fn strided() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().diag(0).unwrap(),
            "Diag([(0, 0), (1, 1), (2, 2)])",
        }
    }

    // Test that `Show` is correct for `MutDiag`
    #[test]
    fn strided_mut() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().diag_mut(0).unwrap(),
            "Diag([(0, 0), (1, 1), (2, 2)])",
        }
    }
}

mod row {
    use linalg::prelude::*;

    // Test that `Show` is correct for `RowVec`
    #[test]
    fn owned() {
        eq! {
            RowVec::from_fn(3, |i| i),
            "Row([0, 1, 2])",
        }
    }

    // Test that `Show` is correct for `Row`
    #[test]
    fn slice() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().t().row(0).unwrap(),
            "Row([(0, 0), (1, 0), (2, 0)])",
        }
    }

    // Test that `Show` is correct for `MutRow`
    #[test]
    fn slice_mut() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().t().row_mut(0).unwrap(),
            "Row([(0, 0), (1, 0), (2, 0)])",
        }
    }

    // Test that `Show` is correct for `strided::Row`
    #[test]
    fn strided() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().row(0).unwrap(),
            "Row([(0, 0), (0, 1), (0, 2)])",
        }
    }

    // Test that `Show` is correct for `strided::MutRow`
    #[test]
    fn strided_mut() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().row_mut(0).unwrap(),
            "Row([(0, 0), (0, 1), (0, 2)])",
        }
    }
}

mod trans {
    use linalg::prelude::*;

    // Test that `Show` is correct for `Trans<Mat>`
    #[test]
    fn mat() {
        eq! {
            Mat::from_fn((2, 2), |i| i).unwrap().t(),
            "[(0, 0), (1, 0)]\n[(0, 1), (1, 1)]",
        }
    }

    // Test that `Show` is correct for `Trans<View>`
    #[test]
    fn view() {
        eq! {
            Mat::from_fn((4, 4), |i| i).unwrap().slice((1, 2), (3, 4)).unwrap().t(),
            "[(1, 2), (2, 2)]\n[(1, 3), (2, 3)]",
        }
    }

    // Test that `Show` is correct for `Trans<MutView>`
    #[test]
    fn view_mut() {
        eq! {
            Mat::from_fn((4, 4), |i| i).unwrap().slice_mut((1, 2), (3, 4)).unwrap().t(),
            "[(1, 2), (2, 2)]\n[(1, 3), (2, 3)]",
        }
    }
}

// Test that `Show` is correct for `Mat`
#[test]
fn mat() {
    eq! {
        Mat::from_fn((2, 2), |i| i).unwrap(),
        "[(0, 0), (0, 1)]\n[(1, 0), (1, 1)]",
    }
}


// Test that `Show` is correct for `View`
#[test]
fn view() {
    eq! {
        Mat::from_fn((4, 4), |i| i).unwrap().slice((1, 2), (3, 4)).unwrap(),
        "[(1, 2), (1, 3)]\n[(2, 2), (2, 3)]",
    }
}

// Test that `Show` is correct for `MutView`
#[test]
fn view_mut() {
    eq! {
        Mat::from_fn((4, 4), |i| i).unwrap().slice_mut((1, 2), (3, 4)).unwrap(),
        "[(1, 2), (1, 3)]\n[(2, 2), (2, 3)]",
    }
}
