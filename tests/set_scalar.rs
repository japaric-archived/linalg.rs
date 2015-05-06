//! Given:
//!
//! - `m = zeros()`
//! - `m.slice_mut(a..b, c..d).set(one())`
//!
//! Test that
//!
//! `if r in a..b && c in c..d { m[r, c] == one() } else { m[r, c] = zero() }`
//!
//! for any valid `a`, `b`, `c`, `d`, `r`, `c`

#![feature(custom_attribute)]
#![feature(plugin)]
#![plugin(quickcheck_macros)]

extern crate complex;
extern crate linalg;
extern crate onezero;
extern crate quickcheck;
extern crate rand;

#[macro_use]
mod setup;

mod transposed {
    mod col {
        use complex::{c64, c128};
        use linalg::prelude::*;
        use onezero::{One, Zero};
        use quickcheck::TestResult;

        macro_rules! tests {
            ($($t:ident),+) => {
                $(
                    #[quickcheck]
                    fn $t(
                        (nrows, ncols): (u32, u32),
                        (srow, scol): (u32, u32),
                        (row, col): (u32, u32),
                        fcol: u32,
                    ) -> TestResult {
                        enforce! {
                            col < scol + nrows,
                            fcol < ncols,
                            row < srow + ncols,
                        }

                        let _0: $t = $t::zero();
                        let _1: $t = $t::one();
                        let mut m = Mat::from_elem((srow + ncols, scol + nrows), _0);
                        m.slice_mut((srow.., scol..)).t().col_mut(fcol).set(_1);

                        if col >= scol && row == srow + fcol {
                            test_eq!(m[(row, col)], _1)
                        } else {
                            test_eq!(m[(row, col)], _0)
                        }
                    }
                 )+
            }
        }

        tests!(f32, f64, c64, c128);
    }

    // TODO
    mod diag {}

    mod row {
        use complex::{c64, c128};
        use linalg::prelude::*;
        use onezero::{One, Zero};
        use quickcheck::TestResult;

        macro_rules! tests {
            ($($t:ident),+) => {
                $(
                    #[quickcheck]
                    fn $t(
                        (nrows, ncols): (u32, u32),
                        (srow, scol): (u32, u32),
                        (row, col): (u32, u32),
                        frow: u32,
                    ) -> TestResult {
                        enforce! {
                            col < scol + nrows,
                            frow < nrows,
                            row < srow + ncols,
                        }

                        let _0: $t = $t::zero();
                        let _1: $t = $t::one();
                        let mut m = Mat::from_elem((srow + ncols, scol + nrows), _0);
                        m.slice_mut((srow.., scol..)).t().row_mut(frow).set(_1);

                        if col == scol + frow && row >= srow {
                            test_eq!(m[(row, col)], _1)
                        } else {
                            test_eq!(m[(row, col)], _0)
                        }
                    }
                 )+
            }
        }

        tests!(f32, f64, c64, c128);
    }

    mod submat {
        use complex::{c64, c128};
        use linalg::prelude::*;
        use onezero::{One, Zero};
        use quickcheck::TestResult;

        macro_rules! tests {
            ($($t:ident),+) => {
                $(
                    #[quickcheck]
                    fn $t(
                        (nrows, ncols): (u32, u32),
                        (srow, scol): (u32, u32),
                        (row, col): (u32, u32),
                    ) -> TestResult {
                        enforce! {
                            col < srow + nrows,
                            row < scol + ncols,
                        }

                        let _0: $t = $t::zero();
                        let _1: $t = $t::one();
                        let mut m = Mat::from_elem((scol + ncols, srow + nrows), _0);
                        m.slice_mut(..).t().slice_mut((srow.., scol..)).set(_1);

                        if row >= scol && col >= srow {
                            test_eq!(m[(row, col)], _1)
                        } else {
                            test_eq!(m[(row, col)], _0)
                        }
                    }
                 )+
            }
        }

        tests!(f32, f64, c64, c128);
    }
}

mod col {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use onezero::{One, Zero};
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (nrows, ncols): (u32, u32),
                    (srow, scol): (u32, u32),
                    (row, col): (u32, u32),
                    fcol: u32,
                ) -> TestResult {
                    enforce! {
                        col < scol + ncols,
                        fcol < ncols,
                        row < srow + nrows,
                    }

                    let _0: $t = $t::zero();
                    let _1: $t = $t::one();
                    let mut m = Mat::from_elem((srow + nrows, scol + ncols), _0);
                    m.slice_mut((srow.., scol..)).col_mut(fcol).set(_1);

                    if col == scol + fcol && row >= srow {
                        test_eq!(m[(row, col)], _1)
                    } else {
                        test_eq!(m[(row, col)], _0)
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

// TODO
mod diag {}

mod row {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use onezero::{One, Zero};
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (nrows, ncols): (u32, u32),
                    (srow, scol): (u32, u32),
                    (row, col): (u32, u32),
                    frow: u32,
                ) -> TestResult {
                    enforce! {
                        col < scol + ncols,
                        frow < nrows,
                        row < srow + nrows,
                    }

                    let _0: $t = $t::zero();
                    let _1: $t = $t::one();
                    let mut m = Mat::from_elem((srow + nrows, scol + ncols), _0);
                    m.slice_mut((srow.., scol..)).row_mut(frow).set(_1);

                    if  col >= scol && row == srow + frow {
                        test_eq!(m[(row, col)], _1)
                    } else {
                        test_eq!(m[(row, col)], _0)
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}

mod submat {
    use complex::{c64, c128};
    use linalg::prelude::*;
    use onezero::{One, Zero};
    use quickcheck::TestResult;

    macro_rules! tests {
        ($($t:ident),+) => {
            $(
                #[quickcheck]
                fn $t(
                    (nrows, ncols): (u32, u32),
                    (srow, scol): (u32, u32),
                    (row, col): (u32, u32),
                ) -> TestResult {
                    enforce! {
                        col < scol + ncols,
                        row < srow + nrows,
                    }

                    let _0: $t = $t::zero();
                    let _1: $t = $t::one();
                    let mut m = Mat::from_elem((srow + nrows, scol + ncols), _0);
                    m.slice_mut((srow.., scol..)).set(_1);

                    if row >= srow && col >= scol {
                        test_eq!(m[(row, col)], _1)
                    } else {
                        test_eq!(m[(row, col)], _0)
                    }
                }
             )+
        }
    }

    tests!(f32, f64, c64, c128);
}
