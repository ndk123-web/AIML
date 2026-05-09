## Methods

- There are nothing but the methods for the initialzation of the weights for each layer neurons to prevent model from going exploiding / vanishing gradient problem
- It uses concept of Normal Distribution for symmetry and randomness and standard derivation to balance the spread of weights

1. Kaiming
   - It basically used for the activation function like ReLu
   - formulas = root(2 / fan_in)

2. Xavier
   - It is used for basically sigmoid and tanh activation functions
   - formulas = root(2 / (fan_in + fan_out))
