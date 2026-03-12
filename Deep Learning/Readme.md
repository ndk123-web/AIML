# Deep Learning

## What is Deep Learning?

Deep Learning is a subset of Machine Learning that uses artificial neural networks with multiple layers (hence "deep") to learn from data. It mimics how the human brain processes information through interconnected neurons.

## Hardware Requirements

**Why GPU/TPU?**
- Deep Learning models require massive parallel computations
- GPUs (Graphics Processing Units) are designed for parallel processing
- TPUs (Tensor Processing Units) are even more specialized for tensor operations
- Training on CPU is possible but extremely slow for large models

**Recommended Hardware:**
- GPU: NVIDIA GPUs with CUDA support (GTX 1060 or higher)
- RAM: Minimum 8GB, recommended 16GB+
- Storage: SSD for faster data loading

## Popular Deep Learning Frameworks

### 1. **TensorFlow** (Google)
- Industry standard for production
- Excellent for deployment
- Built-in Keras API for easy model building
- Great community support and documentation

### 2. **PyTorch** (Facebook/Meta)
- Preferred in research and academia
- More Pythonic and intuitive
- Dynamic computation graphs
- Excellent for experimentation

### 3. **Keras**
- High-level API (now integrated into TensorFlow)
- Very beginner-friendly
- Quick prototyping
- Can run on TensorFlow, CNTK, or Theano backend

### 4. **CNTK** (Microsoft)
- Microsoft's framework
- Good performance
- Less popular than TensorFlow/PyTorch

## Understanding Neurons

### What is a Neuron?

A neuron is the basic unit of a neural network, inspired by biological neurons in the brain.

**Simple Example: Logistic Regression = Single Neuron**

A single neuron performs two steps:

#### Step 1: Linear Combination (Weighted Sum)
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

Where:
- x₁, x₂, ..., xₙ = input features
- w₁, w₂, ..., wₙ = weights (learned parameters)
- b = bias (intercept)
- y = output before activation
```

#### Step 2: Activation Function
```
output = activation(y)

For example, using Sigmoid:
output = 1 / (1 + e^(-y))

This gives us a value between 0 and 1
```

### Neural Network = Multiple Connected Neurons

- **Input Layer**: Receives the raw data
- **Hidden Layers**: Process and transform the data (multiple layers make it "deep")
- **Output Layer**: Produces the final prediction

Each neuron in one layer is connected to neurons in the next layer, and these connections have weights that are learned during training.
