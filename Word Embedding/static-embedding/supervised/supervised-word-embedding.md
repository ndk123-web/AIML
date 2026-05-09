## Supervised Word Embedding

- Here we Decide Vocabulary / Dictionary Size for Our Problem
- We Decide for each Word how many features should be there

- Assume 500 Dictionary is there like (good, bad, awesom, ...)
- and we decided that to represent each word we will initialize with random weights of size n

- Now we run the NN and loss and we optimized the weights for 500 Dictionaries words and in last we get the optimized weights for each word (there will be like
  "good" and "bad" might be having approx same vector size)

## Now what we called when we say there are model with `8 Billion` parameters ?

- Here Parameters are Nothing But Addition per each layer of Weights And Biases
- Example:
  - Assume we have
    1. 500 Dictionary Size with 5 vector features
    2. 2 Hidden Layers with 10 Neurons
    3. Output will be binary (0 or 1)
  - To Calculate our Parameres
    1. Embedding Layer => 500 \* 5 = 2500 weights
    2. Hidden Layer 1 => 5 \* 10 = 50
    3. Hidden Layer 2 => 10 \* 10 = 100

    Total Parameters => 2500 + 50 + 100 = 2650 Parameters (Isn't it easy)
