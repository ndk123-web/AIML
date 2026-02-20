"""
This code demonstrates how to save a trained machine learning model using the `pickle` library in Python. The `pickle` module allows you to serialize and deserialize Python objects, making it easy to save and load models for later use.
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("homeprices.csv")

X = df.drop("price", axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = LinearRegression()
model.fit(X_train,y_train)

ans = model.predict([[1000]])
print(type(ans))
print(ans)

with open("model_joblib", "wb") as f:
    joblib.dump(model, f)

with open("model_joblib", "rb") as f:
    model = joblib.load(f)
    
    ans = model.predict([[1000]])
    print(type(ans))
    print(ans[0])