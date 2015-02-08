#![feature(plugin)]

extern crate linalg;
extern crate quickcheck;
#[plugin]
extern crate quickcheck_macros;
extern crate rand;

use linalg::prelude::*;

#[macro_use]
mod setup;

mod col {
    mod slice {
        use linalg::prelude::*;
        use quickcheck::TestResult;

        use setup;

        // Test that `size_hint()` is correct for `Col::iter()` output
        #[quickcheck]
        fn items((nrows, ncols): (usize, usize), col: usize, skip: usize) -> TestResult {
            enforce! {
                col < ncols,
            }

            test!({
                let m = setup::mat((nrows, ncols));
                let c = try!(m.col(col));
                let n = m.nrows();

                c.iter().skip(skip).size_hint() == if skip > n {
                    (0, Some(0))
                } else {
                    let left = n - skip;

                    (left, Some(left))
                }
            })
        }

        // Test that `size_hint()` is correct for `MutCol::iter_mut()` output
        #[quickcheck]
        fn items_mut((nrows, ncols): (usize, usize), col: usize, skip: usize) -> TestResult {
            enforce! {
                col < ncols,
            }

            test!({
                let mut m = setup::mat((nrows, ncols));
                let n = m.nrows();
                let mut c = try!(m.col_mut(col));

                c.iter_mut().skip(skip).size_hint() == if skip > n {
                    (0, Some(0))
                } else {
                    let left = n - skip;

                    (left, Some(left))
                }
            })
        }
    }

    mod strided {
        use linalg::prelude::*;
        use quickcheck::TestResult;

        use setup;

        // Test that `size_hint()` is correct for `strided::Col::iter()` output
        #[quickcheck]
        fn items((nrows, ncols): (usize, usize), col: usize, skip: usize) -> TestResult {
            enforce! {
                col < ncols,
            }

            test!({
                let m = setup::mat((ncols, nrows)).t();
                let c = try!(m.col(col));
                let n = m.nrows();

                c.iter().skip(skip).size_hint() == if skip > n {
                    (0, Some(0))
                } else {
                    let left = n - skip;

                    (left, Some(left))
                }
            })
        }

        // Test that `size_hint()` is correct for `strided::MutCol::iter_mut()` output
        #[quickcheck]
        fn items_mut((nrows, ncols): (usize, usize), col: usize, skip: usize) -> TestResult {
            enforce! {
                col < ncols,
            }

            test!({
                let mut m = setup::mat((ncols, nrows)).t();
                let n = m.nrows();
                let mut c = try!(m.col_mut(col));

                c.iter_mut().skip(skip).size_hint() == if skip > n {
                    (0, Some(0))
                } else {
                    let left = n - skip;

                    (left, Some(left))
                }
            })
        }
    }
}

mod diag {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `size_hint()` is correct for `Diag:iter()` output
    #[quickcheck]
    fn items(size: (usize, usize), diag: isize, skip: usize) -> TestResult {
        validate_diag!(diag, size);

        test!({
            let m = setup::mat(size);
            let d = try!(m.diag(diag));
            let n = d.len();

            d.iter().skip(skip).size_hint() == if skip > n {
                (0, Some(0))
            } else {
                let left = n - skip;

                (left, Some(left))
            }
        })
    }

    // Test that `size_hint()` is correct for `MutDiag::iter_mut()` output
    #[quickcheck]
    fn items_mut(size: (usize, usize), diag: isize, skip: usize) -> TestResult {
        validate_diag!(diag, size);

        test!({
            let mut m = setup::mat(size);
            let mut d = try!(m.diag_mut(diag));
            let n = d.len();

            d.iter_mut().skip(skip).size_hint() == if skip > n {
                (0, Some(0))
            } else {
                let left = n - skip;

                (left, Some(left))
            }
        })
    }
}

mod row {
    mod slice {
        use linalg::prelude::*;
        use quickcheck::TestResult;

        use setup;

        // Test that `size_hint()` is correct for `Row::iter()` output
        #[quickcheck]
        fn items((nrows, ncols): (usize, usize), row: usize, skip: usize) -> TestResult {
            enforce! {
                row < nrows,
            }

            test!({
                let m = setup::mat((ncols, nrows)).t();
                let r = try!(m.row(row));
                let n = m.ncols();

                r.iter().skip(skip).size_hint() == if skip > n {
                    (0, Some(0))
                } else {
                    let left = n - skip;

                    (left, Some(left))
                }
            })
        }

        // Test that `size_hint()` is correct for `MutRow::iter_mut()` output
        #[quickcheck]
        fn items_mut((nrows, ncols): (usize, usize), row: usize, skip: usize) -> TestResult {
            enforce! {
                row < nrows,
            }

            test!({
                let mut m = setup::mat((ncols, nrows)).t();
                let n = m.ncols();
                let mut r = try!(m.row_mut(row));

                r.iter_mut().skip(skip).size_hint() == if skip > n {
                    (0, Some(0))
                } else {
                    let left = n - skip;

                    (left, Some(left))
                }
            })
        }
    }

    mod strided {
        use linalg::prelude::*;
        use quickcheck::TestResult;

        use setup;

        // Test that `size_hint()` is correct for `strided::Row::iter()` output
        #[quickcheck]
        fn items((nrows, ncols): (usize, usize), row: usize, skip: usize) -> TestResult {
            enforce! {
                row < nrows,
            }

            test!({
                let m = setup::mat((nrows, ncols));
                let r = try!(m.row(row));
                let n = m.ncols();

                r.iter().skip(skip).size_hint() == if skip > n {
                    (0, Some(0))
                } else {
                    let left = n - skip;

                    (left, Some(left))
                }
            })
        }

        // Test that `size_hint()` is correct for `strided::MutRow::iter_mut()` output
        #[quickcheck]
        fn items_mut((nrows, ncols): (usize, usize), row: usize, skip: usize) -> TestResult {
            enforce! {
                row < nrows,
            }

            test!({
                let mut m = setup::mat((nrows, ncols));
                let n = m.ncols();
                let mut r = try!(m.row_mut(row));

                r.iter_mut().skip(skip).size_hint() == if skip > n {
                    (0, Some(0))
                } else {
                    let left = n - skip;

                    (left, Some(left))
                }
            })
        }
    }
}

mod mat {
    use linalg::prelude::*;

    use setup;

    // Test that `size_hint()` is correct for `Mat::iter()` output
    #[quickcheck]
    fn items((nrows, ncols): (usize, usize), skip: usize) -> bool {
        let m = setup::mat((nrows, ncols));
        let n = nrows * ncols;

        m.iter().skip(skip).size_hint() == if skip > n {
            (0, Some(0))
        } else {
            let left = n - skip;

            (left, Some(left))
        }
    }

    // Test that `size_hint()` is correct for `Mat::iter_mut()` output
    #[quickcheck]
    fn items_mut((nrows, ncols): (usize, usize), skip: usize) -> bool {
        let mut m = setup::mat((nrows, ncols));
        let n = nrows * ncols;

        m.iter_mut().skip(skip).size_hint() == if skip > n {
            (0, Some(0))
        } else {
            let left = n - skip;

            (left, Some(left))
        }
    }
}

mod view {
    use linalg::prelude::*;
    use quickcheck::TestResult;

    use setup;

    // Test that `size_hint()` is correct for `Items`
    #[quickcheck]
    fn items(
        start: (usize, usize),
        (nrows, ncols): (usize, usize),
        skip: usize,
    ) -> TestResult {
        let size = (start.0 + nrows, start.1 + ncols);
        test!({
            let m = setup::mat(size);
            let v = try!(m.slice_from(start));
            let n = nrows * ncols;

            v.iter().skip(skip).size_hint() == if skip > n {
                (0, Some(0))
            } else {
                let left = n - skip;

                (left, Some(left))
            }
        })
    }

    // Test that `size_hint()` is correct for `MutItems`
    #[quickcheck]
    fn items_mut(
        start: (usize, usize),
        (nrows, ncols): (usize, usize),
        skip: usize,
    ) -> TestResult {
        let size = (start.0 + nrows, start.1 + ncols);
        test!({
            let mut m = setup::mat(size);
            let mut v = try!(m.slice_from_mut(start));
            let n = nrows * ncols;

            v.iter_mut().skip(skip).size_hint() == if skip > n {
                (0, Some(0))
            } else {
                let left = n - skip;

                (left, Some(left))
            }
        })
    }
}

// Test that `size_hint()` is correct for `Cols`
#[quickcheck]
fn cols(size: (usize, usize), skip: usize) -> bool {
    let m = setup::mat(size);
    let n = m.ncols();

    m.cols().skip(skip).size_hint() == if skip > n {
        (0, Some(0))
    } else {
        let left = n - skip;

        (left, Some(left))
    }
}

// Test that `size_hint()` is correct for `MutCols`
#[quickcheck]
fn mut_cols(size: (usize, usize), skip: usize) -> bool {
    let mut m = setup::mat(size);
    let n = m.ncols();

    m.mut_cols().skip(skip).size_hint() == if skip > n {
        (0, Some(0))
    } else {
        let left = n - skip;

        (left, Some(left))
    }
}

// Test that `size_hint()` is correct for `MutRows`
#[quickcheck]
fn mut_rows((nrows, ncols): (usize, usize), skip: usize) -> bool {
    let mut m = setup::mat((nrows, ncols));
    let n = m.nrows();

    m.mut_rows().skip(skip).size_hint() == if skip > n {
        (0, Some(0))
    } else {
        let left = n - skip;

        (left, Some(left))
    }
}

// Test that `size_hint()` is correct for `Rows`
#[quickcheck]
fn rows((nrows, ncols): (usize, usize), skip: usize) -> bool {
    let m = setup::mat((nrows, ncols));
    let n = m.nrows();
    let left = n - skip;

    m.rows().skip(skip).size_hint() == if skip > n {
        (0, Some(0))
    } else {
        (left, Some(left))
    }
}
