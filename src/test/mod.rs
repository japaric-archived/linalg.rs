mod mat;
mod vec;

fn tol<T: NumCast>() -> T {
    NumCast::from(1e-5_f64).unwrap()
}
