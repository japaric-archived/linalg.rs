# Neural Network

**NOTE** You can compare this implementation to the [one] that used the old version of linalg that
works in the nightly channel. The differences are:

[one]: https://github.com/japaric/linalg_examples/tree/master/nn

- This version has operator sugar, so you can write `a_2[.., 1..] = a_1 * theta_1.t()` instead of
  `a_2.slice_mut((.., 1..)).set(a_1 * theta_1.t())`.

- This version is 2x faster than the other one. About 25% of the speedup comes from changing the
  memory layout of some matrices to row major order (in the old version everything was in column
  major order); the rest of the speed up comes from parallelizing the `g` (sigmoid) and `dgdz`
  (derivative of the sigmoid) functions.

---

Training and validating a neural network (NN) for identification of hand written digits.

This is a classification problem, the input is a 28x28 grayscale image (that depicts a single
digit) and the output is a label (an integer) that ranges from 0 to 9.

The NN is a MLP (multi layer perceptron) with a single hidden layer, trained using batch processing
and gradient descent with adaptive learning rate.

The [MNIST] database was used for training and validation. The database consists of a training set
of 60,000 examples and a test set of 10,000 examples.

[MNIST]: http://yann.lecun.com/exdb/mnist/

To run this example, use the following commands:

(Be sure to have libblas installed)

```
# At the root of the cargo project
# Heads up! This downloads ~10 MB of compressed data, which then gets uncompressed to ~50 MB
$ src/nn/fetch_data.sh
$ cargo run --release
```

Feel free to experiment by changing the "constants" at the top of the `src/main.rs` file.

Here's an example output that uses the whole database.

```
Number of hidden units: 300
Normalization parameter: 0.1
Initial learning rate: 0.3
Momentum: 0.9

Storing a sample of the training set to training_set.png
```

training_set.png

![training set](/src/nn/training_set.png "This is what the training set looks like")

```
The untrained NN classified the first row of the sample as:
[6, 3, 3, 3, 3, 3, 6, 3, 3, 3]

Training the NN with 60000 examples
Epochs MSE    LR
0      15.2534 0.3000
4      5.4466 0.1823
7      2.8247 0.0528
(..)
1114   0.0360 12.7882
1118   0.0356 3.8860
1120   0.0356 2.1422 (local minima)
Training took 2644.718559564 s

The trained NN now classifies the first row of the sample as:
[1, 7, 9, 8, 8, 2, 5, 7, 3, 2]

Validating NN with 10000 examples
Validation took 0.076304605 s

191 of 10000 examples were misclassified (1.91% error rate)

Storing a sample of the misclassified digits to errors.png
```

errors.png

![errors](/src/nn/errors.png "Digits misclassified by the NN, can *you* recongize all of them?")

```
The first row of the sample was misclassified as:
[2, 7, 3, 0, 8, 5, 2, 3, 2, 8]

Correct labels were:
[4, 2, 5, 6, 9, 3, 8, 7, 8, 2]
```
