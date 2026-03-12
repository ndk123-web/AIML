# Cost and Lost Functions

1. Cost Functions
   - Its a function that uses for calculates all the average errors given by the Each Loss Function

2. Loss Functions
   - Its a function that uses for calculates the error for a single training example

- Example:
  - DATA: [1, 2, 3, 4, 5]
  - PREDICTION: [1.1, 2.2, 3.1, 3.9, 5.2]
  - LOSS: [0.1, 0.2, 0.1, 0.1, 0.2]

  - COST: (0.1 + 0.2 + 0.1 + 0.1 + 0.2) / 5 = 0.14

  - now to find the avaerage Cost we have different types of formulas:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - etc.
  - Once i find the cost of all losses thats a one `epoch`
