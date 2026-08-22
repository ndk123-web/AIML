"""
This code demonstrates how to save a trained machine learning model using the `pickle` library in Python. The `pickle` module allows you to serialize and deserialize Python objects, making it easy to save and load models for later use.
"""

import pickle 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("homeprices.csv")

X = df.drop("price", axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = LinearRegression()
model.fit(X_train,y_train)

ans = model.predict([[1000]])
print(type(ans))
print(ans)

# Saving the model using pickle
# with open("model_pickle", "wb") as f: means we are opening a file named "model_pickle" in write-binary mode. The "wb" mode is used because we are writing a binary file, which is necessary for pickled objects.
# pickle.dump(model,f) is the line that actually saves the model. The pickle.dump() function takes two arguments: the object to be pickled (in this case, the trained model) and the file object (f) where the pickled data will be written. This line serializes the model and writes it to the file "model_pickle".
with open("model_pickle", "wb") as f:
    pickle.dump(model,f)
    

# Load pickel Model 
with open("model_pickle", "rb") as f:
    model = pickle.load(f)
    
    ans = model.predict([[1000]])
    print(type(ans))
    print(ans[0])