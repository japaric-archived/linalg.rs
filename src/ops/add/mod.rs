#![doc(hidden)]

// Implement `a + b` as `{ a += b; a }`
macro_rules! assign {
    ($lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, 'c, T> Add<$rhs> for $lhs where $(T: $bound),+ {
                type Output = $lhs;

                fn add(mut self, rhs: $rhs) -> $lhs {
                    self += rhs;
                    self
                }
            }

            impl<'a, 'b, 'c, T> Add<$lhs> for $rhs where $(T:$bound),+ {
                type Output = $lhs;

                fn add(self, mut rhs: $lhs) -> $lhs {
                    rhs += self;
                    rhs
                }
            }
         )+
    };
    (half $lhs:ty { $($rhs:ty { $($bound:ident),+ }),+, }) => {
        $(
            impl<'a, 'b, 'c, T> Add<$rhs> for $lhs where $(T: $bound),+ {
                type Output = $lhs;

                fn add(mut self, rhs: $rhs) -> $lhs {
                    self += rhs;
                    self
                }
            }
         )+
    };
}

mod col;
mod mat;
mod row;
