use std::mem;

use notsafe::UnsafeMatrixMutCol;
use {Col, MutCols};

impl<'a, D, M: UnsafeMatrixMutCol<'a, D>> DoubleEndedIterator<Col<D>> for MutCols<'a, M> {
    fn next_back(&mut self) -> Option<Col<D>> {
        if self.state == self.stop {
            None
        } else {
            self.stop -= 1;
            // XXX This doesn't *really* alias memory, because the columns are not overlapping
            let alias = unsafe { mem::transmute::<_, &mut M>(self.mat as *mut M) };
            Some(unsafe { alias.unsafe_mut_col(self.stop) })
        }
    }
}

impl<'a, D, M: UnsafeMatrixMutCol<'a, D>> Iterator<Col<D>> for MutCols<'a, M> {
    fn next(&mut self) -> Option<Col<D>> {
        if self.state == self.stop {
            None
        } else {
            // XXX This doesn't *really* alias memory, because the columns are not overlapping
            let alias = unsafe { mem::transmute::<_, &mut M>(self.mat as *mut M) };
            let col = unsafe { alias.unsafe_mut_col(self.state) };
            self.state += 1;
            Some(col)
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let exact = self.stop - self.state;
        (exact, Some(exact))
    }
}
