use std::mem;

use notsafe::UnsafeMatrixMutRow;
use {MutRows, Row};

impl<'a, D, M> DoubleEndedIterator<Row<D>> for MutRows<'a, M> where M: UnsafeMatrixMutRow<'a, D> {
    fn next_back(&mut self) -> Option<Row<D>> {
        if self.state == self.stop {
            None
        } else {
            self.stop -= 1;
            // XXX This doesn't *really* alias memory, because the rows are not overlapping
            let alias = unsafe { mem::transmute::<_, &mut M>(self.mat as *mut M) };
            Some(unsafe { alias.unsafe_mut_row(self.stop) })
        }
    }
}

impl<'a, D, M> Iterator<Row<D>> for MutRows<'a, M> where M: UnsafeMatrixMutRow<'a, D> {
    fn next(&mut self) -> Option<Row<D>> {
        if self.state == self.stop {
            None
        } else {
            // XXX This doesn't *really* alias memory, because the rows are not overlapping
            let alias = unsafe { mem::transmute::<_, &mut M>(self.mat as *mut M) };
            let row = unsafe { alias.unsafe_mut_row(self.state) };
            self.state += 1;
            Some(row)
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let exact = self.stop - self.state;
        (exact, Some(exact))
    }
}
