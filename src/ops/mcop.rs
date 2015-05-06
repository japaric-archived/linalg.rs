use std::iter;
use std::ops::{Index, IndexMut, Range};

use extract::Extract;

/// Solves the [Matrix Chain Ordering Problem][MCOP]
///
/// [MCOP]: https://en.wikipedia.org/wiki/Matrix_chain_multiplication
///
/// Given a chain `M = [A, B, C, ..,  Z]` of length `N`, where:
///
/// - `A.size() == (m, n)`
/// - `B.size() == (n, p)`
/// - `C.size() == (p, q)`
/// - ..
///
/// `dims` is an slice that contains the dimensions of the matrices: `[m, n, p, q, ..]`
///
/// It holds that the `i`th matrix `M[i]` has dimensions `(dims[i], dims[i+1])`
///
/// This functions returns a table that contains the optimal "split" indices for each sub-chain.
///
/// For example, `split[0..4] == 2` indicates that:
///
/// The sub-chain `M[0..4] = A * B * C * D` is optimally performed as:
///
/// `M[0..4] = M[0..2] * M[2..4] = (A * B) * (C * D)`
///
/// Another example, `split[2..6] = 3` indicates that:
///
/// `M[2..6] = C * D * E * F` is optimally performed as:
///
/// `M[2..6] = M[2..3] * M[3..6] = C * (D * E * F)`
pub fn solve(dims: &[u64]) -> Table<usize> {
    // Number of matrices
    let n = dims.len() - 1;

    // `cost[i, j]` is the optimal cost of multiplying the sub-chain `M[i..j]`
    let mut cost = Table::from_elem(n, 0);
    let mut split = Table::from_elem(n, 0);

    // `len` is the chain length
    for len in iter::range_inclusive(2, n) {
        // `s` is the start of the chain
        for s in iter::range_inclusive(0, n - len) {
            // `e` is the end of the chain
            let e = s + len;
            cost[s..e] = u64::max_value();

            // `i` is the split index
            for i in s+1..e {
                let q = cost[s..i] + cost[i..e] + dims[s] * dims[i] * dims[e];

                if q < cost[s..e] {
                    cost[s..e] = q;
                    split[s..e] = i;
                }
            }
        }
    }

    split
}

/// A table that collects information about the sub-chains `M[i..j]`
pub struct Table<T> {
    data: Box<[T]>,
    size: usize,
}

impl<T> Table<T> {
    /// Creates a table for all the sub-chains of `M[0..n]`, and initializes all the contents to
    /// `elem`
    fn from_elem(n: usize, elem: T) -> Table<T> where T: Clone {
        let len = (n * (n + 1)) / 2;

        Table {
            data: iter::repeat(elem).take(len).collect::<Vec<_>>().into_boxed_slice(),
            size: n,
        }
    }
}

impl<T> Index<Range<usize>> for Table<T> {
    type Output = T;

    fn index(&self, Range { start: s, end: e }: Range<usize>) -> &T {
        unsafe {
            assert!(e > s);
            assert!(e <= self.size);

            let n = self.size;

            if s < 2 {
                self.data.get(s * (n - 1) + e - 1).extract()
            } else {
                self.data.get(s * (n - 1) + e - (s * (s - 1)) / 2 - 1).extract()
            }
        }
    }
}

impl<T> IndexMut<Range<usize>> for Table<T> {
    fn index_mut(&mut self, r: Range<usize>) -> &mut T {
        unsafe {
            &mut *(self.index(r) as *const T as *mut T)
        }
    }
}
