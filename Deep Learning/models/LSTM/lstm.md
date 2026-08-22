## Basic Working Of LSTM

- LSTM is best RNN model means its better than RNN
- why ?
  - Because it stores better past data than the RNN
- How ?
  - It Used 2 Memory Cells
    1. Cell Memory (Long Short memory)
    2. Hidden Memory (Short memory) (thats what RNN used)
  - It Used 3 Gates
    1. Forget Gate
       - What to foget from `cell memory`
       - `Uses Sigmoid as activation funciton`
    2. Input Gate
       - What to insert into the `cell memory`
       - `Uses Sigmoid as activation funciton`
    3. Outpt Gate
       - What to send to the next layer's `cell memory`
       - `Uses tanh as activation funciton`

#### There are 5 components Gated RNN try to optiimized this using 1 memory cell and 2 gates only
