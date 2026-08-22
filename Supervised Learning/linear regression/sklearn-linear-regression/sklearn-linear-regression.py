import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 

# Load the dataset
df = pd.read_csv('homeprices.csv')
print(df)

X = df.drop('price' , axis = 'columns')
y = df.price 

# Split the dataset into training and testing sets
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

# Create an instance of the Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)

# Predict the prices for the test set
y_pred = model.predict(X_test)

# Print the predicted prices
print("Predicted prices:", y_pred)

# Print the actual prices
print("Actual prices:", y_test.values)

# Calculate the R-squared score
r_squared = model.score(X_test, y_test)
print("R-squared score:", r_squared)