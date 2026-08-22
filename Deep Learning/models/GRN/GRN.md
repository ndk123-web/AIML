## Basic Working Of GRN

- GRN is best LSTM model means its better than LSTM
- why ?
  - Because it uses less components than the LSTM
- How ?
  - It Used 1 Memory Cells
    1. Hidden Memory (Short Memory (RNN USED) + Long Short memory(LSTM USed))
  - It Used 3 Gates
    1. Update Gate
       - It works as `Input and Forget Gate`
       - `Uses Sigmoid as activation funciton`
    2. Reset Gate
       - It works as `Output Gate`
       - `Uses tanh as activation funciton`
