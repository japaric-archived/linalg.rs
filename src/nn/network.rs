extern crate time;

use std::{cmp, thread};

use cast::From;

// linalg
use ::Buffer;
use ::prelude::*;

use nn::{F, NUM_CPUS};

type cMat = ::Mat<F, ::order::Col>;
type rMat = ::Mat<F, ::order::Row>;

use super::labels::Labels;

use super::rand::{Rng, XorShiftRng, self};

pub struct Options {
    pub epochs: u32,
    /// Normalization parameter (weight decay)
    pub lambda: F,
    pub learning_rate: F,
    pub momentum: F,
}

/// A neural network with a single hidden layer
pub struct Network {
    /// Number of units per layer
    s: (u32, u32, u32),
    /// Parameters
    theta: Box<[F]>,
}

impl Network {
    /// New untrained network
    pub fn new(num_inputs: u32, num_hidden_units: u32, num_outputs: u32) -> Network {
        assert!(num_hidden_units > 0);
        assert!(num_inputs > 0);
        assert!(num_outputs > 0);

        let i = num_inputs;
        let h = num_hidden_units;
        let o = num_outputs;

        let n = usize::from(i + 1) * usize::from(h) + usize::from(h + 1) * usize::from(o);

        let ref mut rng: XorShiftRng = rand::thread_rng().gen();

        // NB Use the unseeded RNG for benchmarks
        //let ref mut rng: XorShiftRng = XorShiftRng::new_unseeded();

        let theta: Vec<_> = (0..n).map(|_| rng.gen::<F>() - 0.5).collect();

        Network {
            s: (i, h, o),
            theta: theta.into_boxed_slice(),
        }
    }

    /// Classifies an image
    pub fn classify(&self, image: &Row<F>) -> u32 {
        let (i, h, o) = self.s;

        assert_eq!(image.len(), i + 1);

        let (theta_1, theta_2) = self.theta();
        let mut a_2 = Row::ones(h + 1);
        let mut h = &mut *Row::zeros(o);

        // Feed forward
        let a_1 = image;

        // a_2 = [1, g(a_1 * theta_1')]
        a_2[1..] = a_1 * theta_1.t();
        g(&mut a_2[1..]);

        // h = g(a_2 * theta_2')
        h[..] = &*a_2 * theta_2.t();
        g(h);

        let mut iter = h.iter().zip(0..);
        let (mut max, mut i) = iter.next().unwrap();

        for (x, j) in iter {
            if x > max {
                max = x;
                i = j;
            }
        }

        i
    }

    /// Trains the neural network
    pub fn train(
        &mut self,
        images: &rMat,
        labels: &rMat,
        options: Options)
    {
        const TOL: F = 1e-5;
        const TICK: u64 = 5_000_000_000;

        assert!(images[.., 0].iter().all(|&x| x == 1.));

        let mut alpha = options.learning_rate;
        let epochs = options.epochs;
        let lambda = options.lambda;
        let m = options.momentum;
        let s = self.s;
        let x = images;
        let y = labels;

        let ref mut buffer = Buffer::new({
            let (i, h, o) = s;
            let m = y.nrows();

            usize::from(m * (3 * h + 2 * o + 1) + 3 * o)
        });

        let (theta_1, theta_2) = self.theta_mut();

        let mut prev_step_1 = cMat::zeros(theta_1.size());
        let mut prev_step_2 = cMat::zeros(theta_2.size());
        let mut prev_theta_1 = cMat::zeros(theta_1.size());
        let mut prev_theta_2 = cMat::zeros(theta_2.size());
        let grad_1 = &mut *cMat::zeros(theta_1.size());
        let grad_2 = &mut *cMat::zeros(theta_2.size());

        let mut last = time::precise_time_ns().wrapping_sub(TICK);
        println!("Epochs MSE    LR");
        for i in 0..epochs {
            let cost = cost_plus_grad(theta_1, theta_2, x, y, s, lambda, grad_1, grad_2, buffer);

            let now = time::precise_time_ns();

            if now.wrapping_sub(last) > TICK {
                last = now;
                println!("{:<6} {:<6.4} {:.4}", i, cost, alpha);
            }

            prev_theta_1[..] = &*theta_1;
            prev_theta_2[..] = &*theta_2;

            loop {
                *theta_1 -= &*grad_1 * alpha;
                *theta_1 += &*prev_step_1 * m;

                *theta_2 -= &*grad_2 * alpha;
                *theta_2 += &*prev_step_2 * m;

                let new_cost = self::cost(theta_1, theta_2, x, y, s, lambda, buffer);

                if (new_cost - cost).abs() / new_cost.max(cost) < TOL {
                    return println!("{:<6} {:<6.4} {:.4} (local minima)", i, cost, alpha);
                } else if new_cost < cost {
                    // Accelerate
                    alpha += 0.05 * alpha;

                    break
                } else {
                    // Rollback
                    alpha /= 2.;

                    theta_1[..] = &*prev_theta_1;
                    theta_2[..] = &*prev_theta_2;

                    prev_step_1[..] = 0.;
                    prev_step_2[..] = 0.;
                }
            }

            prev_step_1[..] = &*theta_1;
            *prev_step_1 -= &*prev_theta_1;

            prev_step_2[..] = &*theta_2;
            *prev_step_2 -= &*prev_theta_2;
        }
    }

    /// Validates the NN, returns a collection of the mismatched examples
    pub fn validate(
        &self,
        images: &rMat,
        labels: &Labels,
    ) -> Vec<u32> {
        let (_, h, o) = self.s;
        let m = images.nrows();

        let (theta_1, theta_2) = self.theta();
        let a_2 = &mut *cMat::ones((m, (h + 1)));
        let h = &mut *rMat::zeros((m, o));

        // Feed forward
        let a_1 = images;

        // a_2 = [ones(m, 1), g(a_1 * theta_1')]
        a_2[.., 1..] = a_1 * theta_1.t();
        g(&mut a_2[.., 1..]);

        // h = g(a_2 * theta_2)
        h[..] = &*a_2 * theta_2.t();
        g(h);

        h.rows().zip(0..).filter_map(|(h, i)| {
            let max = h[u32::from(labels[i])];

            if h.iter().all(|&x| x <= max) {
                None
            } else {
                Some(i)
            }

        }).collect()
    }

    fn theta(&self) -> (&cMat, &cMat) {
        let (i, h, o) = self.s;
        let at = usize::from(i + 1) * usize::from(h);
        let (left, right) = self.theta.split_at(at);

        (cMat::reshape(left, (h, i + 1)), cMat::reshape(right, (o, h + 1)))
    }

    fn theta_mut(&mut self) -> (&mut cMat, &mut cMat) {
        let (i, h, o) = self.s;
        let at = usize::from(i + 1) * usize::from(h);
        let (left, right) = self.theta.split_at_mut(at);

        (cMat::reshape_mut(left, (h, i + 1)), cMat::reshape_mut(right, (o, h + 1)))
    }
}

// TODO Merge `cost` and `cost_plus_grad`
/// Evaluates cost function
fn cost(
    theta_1: &cMat,
    theta_2: &cMat,
    x: &rMat,
    y: &rMat,
    s: (u32, u32, u32),
    lambda: F,
    buffer: &mut Buffer<F>,
) -> F {
    let (_, h, o) = s;
    let m = y.nrows();

    let mut pool = buffer.as_pool();

    let a_2: &mut cMat = pool.mat((m, h + 1));
    let a_3: &mut rMat = pool.mat((m, o));

    // Feed forward
    debug_assert!(x[.., 0].iter().all(|&x| x == 1.));

    let a_1 = x;

    // a_2 = [ones(m, 1), g(a_1 * theta_1')]
    a_2[.., 0] = 1.;
    a_2[.., 1..] = a_1 * theta_1.t();
    g(&mut a_2[.., 1..]);

    // h = a_3 = g(a_2 * theta_2)
    a_3[..] = &*a_2 * theta_2.t();
    g(a_3);

    // unnormalized_cost = sum(-y .* log(h) - (1 - y) .* log(1 - h)) / m
    let mut cost = 0.;

    let _1_y = pool.row(o);
    let log_h = pool.col(o);
    let log_1_h = pool.col(o);

    for (h, y) in a_3.rows().zip(y.rows()) {
        let h = h.t();

        // 1 - y
        for (_1_y, &y) in _1_y.iter_mut().zip(y) {
            *_1_y = 1. - y;
        }

        // log(h)
        for (log_h, h) in log_h.iter_mut().zip(h) {
            *log_h = h.ln();
        }

        // log(1 - h)
        for (log_1_h, &h) in log_1_h.iter_mut().zip(h) {
            *log_1_h = (1. - h).ln();
        }

        cost -= y * &*log_h + &*_1_y * &*log_1_h;
    }

    // Normalization
    // cost = unnormalized_cost + lambda * (ssq(theta_1[:, 1:]) + ssq(theta_2[:, 1:])) / 2 / m
    let ssq = |m: &cMat| {
        let norm = m[.., 1..].norm();
        norm * norm
    };

    let ssq = ssq(theta_1) + ssq(theta_2);

    cost += lambda * ssq / 2.;
    cost /= F::from(m);

    cost
}

/// Evaluates cost function and its gradients
fn cost_plus_grad(
    theta_1: &cMat,     // Input. (h, i + 1)
    theta_2: &cMat,     // Input. (o, h + 1)
    x: &rMat,           // Input: (m, i + 1)
    y: &rMat,           // Input: (m, o)
    s: (u32, u32, u32),
    lambda: F,
    grad_1: &mut cMat,  // Output. (h, i + 1)
    grad_2: &mut cMat,  // Output. (o, h + 1)
    buffer: &mut Buffer<F>,
) -> F {
    let (_, h, o) = s;
    let m = y.nrows();

    let mut pool = buffer.as_pool();

    let a_2: &mut cMat = pool.mat((m, h + 1));
    let a_3: &mut rMat = pool.mat((m, o));
    let z_2: &mut cMat = pool.mat((m, h));

    let delta_2: &mut cMat = pool.mat((m, h));
    let delta_3: &mut rMat = pool.mat((m, o));

    // Feed forward
    debug_assert!(x[.., 0].iter().all(|&x| x == 1.));

    let a_1 = x;
    let m = F::from(m);

    // z_2 = a_1 * theta_1'
    z_2[..] = a_1 * theta_1.t();

    // a_2 = [ones(m, 1), g(z_2)]
    a_2[.., 0] = 1.;
    a_2[.., 1..] = &*z_2;
    g(&mut a_2[.., 1..]);

    // h = a_3 = g(a_2 * theta_2)
    a_3[..] = &*a_2 * theta_2.t();
    g(a_3);

    // unnormalized_cost = sum(-y .* log(h) - (1 - y) .* log(1 - h)) / m
    let mut cost = 0.;

    let _1_y = pool.row(o);
    let log_h = pool.col(o);
    let log_1_h = pool.col(o);

    for (h, y) in a_3.rows().zip(y.rows()) {
        let h = h.t();

        // 1 - y
        for (_1_y, &y) in _1_y.iter_mut().zip(y) {
            *_1_y = 1. - y;
        }

        // log(h)
        for (log_h, h) in log_h.iter_mut().zip(h) {
            *log_h = h.ln();
        }

        // log(1 - h)
        for (log_1_h, &h) in log_1_h.iter_mut().zip(h) {
            *log_1_h = (1. - h).ln();
        }

        cost -= y * &*log_h + &*_1_y * &*log_1_h;
    }

    // Normalization
    // cost = unnormalized_cost + lambda * (ssq(theta_1[:, 1:]) + ssq(theta_2[:, 1:])) / 2 / m
    let ssq = |m: &cMat| {
        let norm = m[.., 1..].norm();
        norm * norm
    };

    let ssq = ssq(theta_1) + ssq(theta_2);

    cost += lambda * ssq / 2.;
    cost /= m;

    // Back propagation
    // delta_3 = a_3 - y
    delta_3[..] = &*a_3 - y;

    //// D_2 = (delta_3.t() * a_2) / m
    grad_2[..] = (&*delta_3).t() * &*a_2;

    // delta_2 = delta_3 * theta_2[:, 1:] .* g'(z_2)
    let mut dgdz_z_2 = z_2;
    dgdz(dgdz_z_2);
    delta_2[..] = &*delta_3 * &theta_2[.., 1..];
    *delta_2 *= &*dgdz_z_2;

    // D_1 = (delta_2.t() * a_1) / m
    grad_1[..] = (&*delta_2).t() * a_1;

    // Normalization
    // D_i[:, 1:] += lambda * theta_1[:, 1:] / m
    grad_1[.., 1..] += &theta_1[.., 1..] * lambda;
    *grad_1 /= m;

    grad_2[.., 1..] += &theta_2[.., 1..] * lambda;
    *grad_2 /= m;

    cost
}

/// Sigmoid function
fn g<Z: ?Sized>(z: &mut Z) where Z: AsMut<[F]> {
    fn g_(z: &mut [F]) {
        let len = z.len();

        if len > 1_000_000 && NUM_CPUS > 1 {
            let n = cmp::min(len / 1_000_000, NUM_CPUS);
            let sz = (len - 1) / n + 1;

            z.chunks_mut(sz).map(|z| thread::scoped(|| {
                for x in z {
                    *x = 1. / (1. + (-*x).exp())
                }
            })).collect::<Vec<_>>();
        } else {
            for x in z {
                *x = 1. / (1. + (-*x).exp())
            }
        }
    }

    g_(z.as_mut())
}

/// Gradient of the sigmoid function
fn dgdz<Z: ?Sized>(z: &mut Z) where Z: AsMut<[F]> {
    fn dgdz_(z: &mut [F]) {
        let len = z.len();

        if len > 1_000_000 && NUM_CPUS > 1 {
            let n = cmp::min(len / 1_000_000, NUM_CPUS);
            let sz = (len - 1) / n + 1;

            z.chunks_mut(sz).map(|z| thread::scoped(|| {
                for x in z {
                    let g = 1. / (1. + (-*x).exp());
                    *x = g * (1. - g)
                }
            })).collect::<Vec<_>>();
        } else {
            for x in z {
                let g = 1. / (1. + (-*x).exp());
                *x = g * (1. - g)
            }
        }
    }

    dgdz_(z.as_mut())
}
