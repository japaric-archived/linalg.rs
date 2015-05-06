//! Test that
//!
//! - `fmt!("{:?}", mat)` == "[0, 1, 2]\n[3, 4, 5]", etc

extern crate linalg;
extern crate rand;

mod setup;

use linalg::prelude::*;

macro_rules! eq {
    ($lhs:expr, $rhs:expr,) => {
        assert_eq!(format!("{:?}", $lhs), $rhs)
    };
}

mod col {
    use linalg::prelude::*;

    #[test]
    fn contiguous() {
        eq! {
            ::setup::mat((3, 3)).col(0),
            "Col([(0, 0), (1, 0), (2, 0)])",
        }
    }

    #[test]
    fn strided() {
        eq! {
            ::setup::mat((3, 3)).t().col(0),
            "Col([(0, 0), (0, 1), (0, 2)])",
        }
    }
}

mod row {
    use linalg::prelude::*;

    #[test]
    fn contiguous() {
        eq! {
            ::setup::mat((3, 3)).t().row(0),
            "Row([(0, 0), (1, 0), (2, 0)])",
        }
    }
    #[test]
    fn strided() {
        eq! {
            ::setup::mat((3, 3)).row(0),
            "Row([(0, 0), (0, 1), (0, 2)])",
        }
    }
}

mod transposed {
    use linalg::prelude::*;

    #[test]
    fn submat() {
        eq! {
            ::setup::mat((4, 4)).slice((1..3, 2..4)).t(),
            "[(1, 2), (2, 2)]\n[(1, 3), (2, 3)]",
        }
    }
}

#[test]
fn diag() {
    eq! {
        ::setup::mat((3, 3)).diag(0),
        "Diag([(0, 0), (1, 1), (2, 2)])",
    }
}

#[test]
fn submat() {
    eq! {
        setup::mat((4, 4)).slice((1..3, 2..4)),
        "[(1, 2), (1, 3)]\n[(2, 2), (2, 3)]",
    }
}
