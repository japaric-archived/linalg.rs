use blas::Transpose;

use traits::{Matrix, self};
use {Chain, SubMat};

impl<'a, T> Matrix for Chain<'a, T> {
    type Elem = T;

    fn nrows(&self) -> u32 {
        let (trans, mat) = self.first;

        match trans {
            Transpose::No => mat.nrows(),
            Transpose::Yes => mat.ncols(),
        }
    }

    fn ncols(&self) -> u32 {
        let &(trans, mat) = self.tail.last().unwrap_or(&self.second);

        match trans {
            Transpose::No => mat.ncols(),
            Transpose::Yes => mat.nrows(),
        }
    }
}

impl<'a, T> traits::Transpose for Chain<'a, T> {
    type Output = Chain<'a, T>;

    fn t(self) -> Chain<'a, T> {
        fn t<T>((trans, a): (Transpose, SubMat<T>)) -> (Transpose, SubMat<T>) {
            match trans {
                Transpose::No => (Transpose::Yes, a),
                Transpose::Yes => (Transpose::No, a),
            }
        }

        let was_first = t(self.first);
        let was_second = t(self.second);

        match &self.tail[..] {
            [] => Chain { first: was_second, second: was_first, tail: vec![] },
            [last] => Chain { first: t(last), second: was_second, tail: vec![was_first] },
            [head.., second_to_last, last] => {
                let mut tail = Vec::with_capacity(self.tail.len());

                tail.extend(head.iter().rev().map(|&x| t(x)));

                tail.push(was_second);
                tail.push(was_first);

                Chain {
                    first: t(last),
                    second: t(second_to_last),
                    tail: tail,
                }
            },
        }
    }
}
