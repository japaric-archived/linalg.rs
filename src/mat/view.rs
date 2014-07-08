use array::traits::ArrayShape;
use mat::traits::{MatrixCol,MatrixColIterator,MatrixDiag,MatrixRow,
                  MatrixRowIterator, MatrixShape,MatrixView};
use mat::{Col,Cols,Diag,Row,Rows};
use traits::UnsafeIndex;

// XXX I'd prefer to use Sub, but it's taken by the subtraction operator
pub struct View<M> {
    mat: M,
    start: (uint, uint),
    stop: (uint, uint),
}

impl<
    M
> View<M> {
    #[inline]
    pub unsafe fn unsafe_new(mat: M, start: (uint, uint), stop: (uint, uint))
        -> View<M>
    {
        View {
            mat: mat,
            start: start,
            stop: stop,
        }
    }

    #[inline]
    pub fn start(&self) -> (uint, uint) {
        self.start
    }
}

impl<
    M: Copy
> View<M> {
    #[inline]
    pub fn get_ref(&self) -> M {
        self.mat
    }
}

impl<
    M: Copy + MatrixShape
> View<M> {
    #[inline]
    pub fn new(mat: M,
               start@(start_row, start_col): (uint, uint),
               stop@(stop_row, stop_col): (uint, uint))
        -> View<M>
    {
        let (nrows, ncols) = mat.shape();

        assert!(stop_row <= nrows && stop_col <= ncols,
                "view: out of bounds: {}:{} of {}", start, stop, mat.shape());

        assert!(start_row <= stop_row && start_col <= stop_col,
                "view: invalid indexing: {}:{}", start, stop);

        unsafe {
            View::unsafe_new(mat, start, stop)
        }
    }
}

// ArrayShape
impl<
    M: Copy
> ArrayShape<(uint, uint)>
for View<M> {
    #[inline]
    fn shape(&self) -> (uint, uint) {
        let (start_row, start_col) = self.start;
        let (stop_row, stop_col) = self.stop;

        let nrows = stop_row - start_row;
        let ncols = stop_col - start_col;

        (nrows, ncols)
    }
}

// Collection
impl<
    M: Copy
> Collection
for View<M> {
    #[inline]
    fn len(&self) -> uint {
        let (nrows, ncols) = self.shape();

        nrows * ncols
    }
}

// Index
impl<
    T,
    M: Copy + UnsafeIndex<(uint, uint), T>
> Index<(uint, uint), T>
for View<M> {
    #[inline]
    fn index<'a>(&'a self, index@&(row, col): &(uint, uint)) -> &'a T {
        let shape@(nrows, ncols) = self.shape();

        assert!(row < nrows && col < ncols,
                "index: out of bounds: {} of {}", index, shape)

        unsafe { self.unsafe_index(index) }
    }
}

// MatrixCol
impl<
    M: Copy + MatrixShape
> MatrixCol
for View<M> {
    fn col(self, col: uint) -> Col<View<M>> {
        Col::new(self, col)
    }

    unsafe fn unsafe_col(self, col: uint) -> Col<View<M>> {
        Col::unsafe_new(self, col)
    }
}

// MatrixColIterator
impl<
    M: Copy + MatrixShape
> MatrixColIterator
for View<M> {
    #[inline]
    fn cols(self) -> Cols<View<M>> {
        Cols::new(self)
    }
}

// MatrixDiag
impl<
    M: Copy + MatrixShape
> MatrixDiag
for View<M> {
    fn diag(self, diag: int) -> Diag<View<M>> {
        Diag::new(self, diag)
    }
}

// MatrixRow
impl<
    M: Copy + MatrixShape
> MatrixRow
for View<M> {
    fn row(self, row: uint) -> Row<View<M>> {
        Row::new(self, row)
    }

    unsafe fn unsafe_row(self, row: uint) -> Row<View<M>> {
        Row::unsafe_new(self, row)
    }
}

// MatrixRowIterator
impl<
    M: Copy + MatrixShape
> MatrixRowIterator
for View<M> {
    #[inline]
    fn rows(self) -> Rows<View<M>> {
        Rows::new(self)
    }
}

// MatrixShape
impl<
    M: Copy + ArrayShape<(uint, uint)>
> MatrixShape
for View<M> {
}

// MatrixView
impl<
    M: Copy + MatrixShape
> MatrixView<View<M>>
for View<M> {
    fn view(self,
            start@(start_row, start_col): (uint, uint),
            stop@(stop_row, stop_col): (uint, uint))
        -> View<M>
    {
        let shape@(nrows, ncols) = self.shape();

        assert!(stop_row <= nrows && stop_col <= ncols,
                "view: out of bounds: {}:{} of {}", start, stop, shape);

        assert!(start_row <= stop_row && start_col <= stop_col,
                "view: invalid indexing: {}:{}", start, stop);

        unsafe {
            self.unsafe_view(start, stop)
        }
    }

    unsafe fn unsafe_view(self,
                          (start_row, start_col): (uint, uint),
                          (stop_row, stop_col): (uint, uint))
        -> View<M>
    {
        let (offset_row, offset_col) = self.start;

        View::unsafe_new(self.mat,
                         (offset_row + start_row, offset_col + start_col),
                         (offset_row + stop_row, offset_col + stop_col))
    }
}

// UnsafeIndex
impl<
    T,
    M: UnsafeIndex<(uint, uint), T>
> UnsafeIndex<(uint, uint), T>
for View<M> {
    #[inline]
    unsafe fn unsafe_index<'a>(&'a self, &(row, col): &(uint, uint)) -> &'a T {
        let (start_row, start_col) = self.start;

        self.mat.unsafe_index(&(row + start_row, col + start_col))
    }
}
