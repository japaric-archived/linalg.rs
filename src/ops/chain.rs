use ops::Chain;
use traits::{Matrix, Transpose};

impl<'a, T> Matrix for Chain<'a, T> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        self.first.nrows()
    }

    fn ncols(&self) -> u32 {
        self.tail.last().unwrap_or(&self.second).ncols()
    }
}

impl<'a, T> Transpose for Chain<'a, T> {
    type Output = Chain<'a, T>;

    fn t(self) -> Chain<'a, T> {
        let Chain { first, second, tail } = self;
        let was_first = first.t();
        let was_second = second.t();

        match &tail[..] {
            [] => Chain { first: was_second, second: was_first, tail: vec![] },
            [last] => {
                Chain { first: last.t(), second: was_second, tail: vec![was_first] }
            },
            [head.., second_to_last, last] => {
                let mut tail = Vec::with_capacity(tail.len());

                tail.extend(head.iter().rev().map(|x| x.t()));
                tail.push(was_second);
                tail.push(was_first);

                Chain { first: last.t(), second: second_to_last.t(), tail: tail }
            },
        }
    }
}
