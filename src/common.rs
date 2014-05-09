use std::mem::replace;

pub struct Stride<'a, T> {
    slice: &'a [T],
    state: uint,
    step: uint,
    stop: uint,
}

impl<
    'a,
    T
> Stride<'a, T> {
    pub fn new(slice: &'a [T],
               start: uint,
               stop: uint,
               step: uint)
               -> Stride<'a, T> {
        Stride {
            slice: slice,
            state: start,
            step: step,
            stop: stop,
        }
    }
}

impl<
    'a,
    T
> Iterator<&'a T>
for Stride<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        if self.state < self.stop {
            Some(unsafe {
                self.slice.unsafe_ref(replace(&mut self.state,
                                              self.state + self.step))
            })
        } else {
            None
        }
    }
}