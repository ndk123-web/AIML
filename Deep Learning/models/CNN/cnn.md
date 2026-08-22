## CNN

1. CNN just use for detecting objects in image nothing else
2. After CNN detects objects, it gives us matrix of probabilities
3. now taking that matrix we feed that matrix to the Neural Network who already trained for the classification task

## Issue Of Compuation

- if an image of 1920\*1080 and has 2 hidden layers of 1000 neurons each then it will take too much time to train
- To Reduce the Matrix we use Pooling Layer
  1. Max Pooling
     - it takes the maximum value from the next 2\*2 matrix
     - reduces the matrix size to 1/4th
     - reduce overfitting
  2. Average Pooling
     - it takes the average value from the next 2\*2 matrix
     - reduces the matrix size to 1/4th
     - reduce overfitting

## In Short

- We create an CNN algorithm who can detect objects in image (using Convolution + ReLU + Pooling)
- We create an Neural Network who can classify the objects (using Neural Network)
- We feed the output of CNN to the Neural Network

## Actual Flow

Input Image -> Convolution Layer (Feature Applies to image to extract features) -> pooling -> ANN -> Output
 