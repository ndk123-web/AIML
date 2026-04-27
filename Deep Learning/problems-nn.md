# Common Problems in Neural Networks

## 1. Vanishing Gradient Problem
This happens when a neural network has many hidden layers. During training, we use the **Chain Rule** to calculate gradients by multiplying several values together.
- If these values are very small (e.g., less than 1), multiplying them repeatedly makes the final gradient extremely small (close to zero).
- **Result:** The weights in the early layers don't change much, and the network stops learning.

## 2. Exploding Gradient Problem
This is the opposite of the vanishing gradient problem.
- If the values being multiplied in the chain rule are large (e.g., much greater than 1), the final gradient becomes huge.
- **Result:** This leads to very large updates to the weights, making the training unstable and causing the model to "explode" or fail to converge.