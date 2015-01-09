#![allow(unstable)]

extern crate linalg;

use linalg::prelude::*;

macro_rules! eq {
    ($lhs:expr, $rhs:expr,) => { assert_eq!(format!("{:?}", $lhs), $rhs) }
}

mod col {
    use linalg::prelude::*;

    // Test that `Show` is correct for `ColVec`
    #[test]
    fn owned() {
        eq! {
            ColVec::from_fn(3, |i| i),
            "Col([0u, 1u, 2u])",
        }
    }

    // Test that `Show` is correct for `Col`
    #[test]
    fn slice() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().col(0).unwrap(),
            "Col([(0u, 0u), (1u, 0u), (2u, 0u)])",
        }
    }

    // Test that `Show` is correct for `MutCol`
    #[test]
    fn slice_mut() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().col_mut(0).unwrap(),
            "Col([(0u, 0u), (1u, 0u), (2u, 0u)])",
        }
    }

    // Test that `Show` is correct for `strided::Col`
    #[test]
    fn strided() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().t().col(0).unwrap(),
            "Col([(0u, 0u), (0u, 1u), (0u, 2u)])",
        }
    }

    // Test that `Show` is correct for `strided::MutCol`
    #[test]
    fn strided_mut() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().t().col_mut(0).unwrap(),
            "Col([(0u, 0u), (0u, 1u), (0u, 2u)])",
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
            "Diag([(0u, 0u), (1u, 1u), (2u, 2u)])",
        }
    }

    // Test that `Show` is correct for `MutDiag`
    #[test]
    fn strided_mut() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().diag_mut(0).unwrap(),
            "Diag([(0u, 0u), (1u, 1u), (2u, 2u)])",
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
            "Row([0u, 1u, 2u])",
        }
    }

    // Test that `Show` is correct for `Row`
    #[test]
    fn slice() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().t().row(0).unwrap(),
            "Row([(0u, 0u), (1u, 0u), (2u, 0u)])",
        }
    }

    // Test that `Show` is correct for `MutRow`
    #[test]
    fn slice_mut() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().t().row_mut(0).unwrap(),
            "Row([(0u, 0u), (1u, 0u), (2u, 0u)])",
        }
    }

    // Test that `Show` is correct for `strided::Row`
    #[test]
    fn strided() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().row(0).unwrap(),
            "Row([(0u, 0u), (0u, 1u), (0u, 2u)])",
        }
    }

    // Test that `Show` is correct for `strided::MutRow`
    #[test]
    fn strided_mut() {
        eq! {
            Mat::from_fn((3, 3), |i| i).unwrap().row_mut(0).unwrap(),
            "Row([(0u, 0u), (0u, 1u), (0u, 2u)])",
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
            "[(0u, 0u), (1u, 0u)]\n[(0u, 1u), (1u, 1u)]",
        }
    }

    // Test that `Show` is correct for `Trans<View>`
    #[test]
    fn view() {
        eq! {
            Mat::from_fn((4, 4), |i| i).unwrap().slice((1, 2), (3, 4)).unwrap().t(),
            "[(1u, 2u), (2u, 2u)]\n[(1u, 3u), (2u, 3u)]",
        }
    }

    // Test that `Show` is correct for `Trans<MutView>`
    #[test]
    fn view_mut() {
        eq! {
            Mat::from_fn((4, 4), |i| i).unwrap().slice_mut((1, 2), (3, 4)).unwrap().t(),
            "[(1u, 2u), (2u, 2u)]\n[(1u, 3u), (2u, 3u)]",
        }
    }
}

// Test that `Show` is correct for `Mat`
#[test]
fn mat() {
    eq! {
        Mat::from_fn((2, 2), |i| i).unwrap(),
        "[(0u, 0u), (0u, 1u)]\n[(1u, 0u), (1u, 1u)]",
    }
}


// Test that `Show` is correct for `View`
#[test]
fn view() {
    eq! {
        Mat::from_fn((4, 4), |i| i).unwrap().slice((1, 2), (3, 4)).unwrap(),
        "[(1u, 2u), (1u, 3u)]\n[(2u, 2u), (2u, 3u)]",
    }
}

// Test that `Show` is correct for `MutView`
#[test]
fn view_mut() {
    eq! {
        Mat::from_fn((4, 4), |i| i).unwrap().slice_mut((1, 2), (3, 4)).unwrap(),
        "[(1u, 2u), (1u, 3u)]\n[(2u, 2u), (2u, 3u)]",
    }
}
