#![doc(hidden)]

// Implements `a - b` as `{ a -= b; a }`
macro_rules! assign {
    ($lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, 'c, T> Sub<$rhs> for $lhs where T: Neg<Output=T>, $(T: $bound),+ {
                type Output = $lhs;

                fn sub(mut self, rhs: $rhs) -> $lhs {
                    self -= rhs;
                    self
                }
            }
         )+
    };
}

mod col;
mod mat;
mod row;
