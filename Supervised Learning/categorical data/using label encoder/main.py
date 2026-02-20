from sklearn.preprocessing import LabelEncoder

# write label encoder code here
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Example dataset
df = pd.DataFrame({
    "area": [1000, 1500, 1200, 1800],
    "bedrooms": [2, 3, 2, 4],
    "town": ["Delhi", "Mumbai", "Delhi", "Pune"],
    "price": [100, 200, 120, 300]
})

# Separate features and target
X = df.drop("price", axis=1)
y = df["price"]

# Encode categorical column
le = LabelEncoder()
X["town"] = le.fit_transform(X["town"])

print(X)

model = LinearRegression()
model.fit(X, y)

new_data = pd.DataFrame({
    "area": [1000], 
    "bedrooms": [2],
    "town": le.transform(["Mumbai"]) # we have to transform the new data using the
    # same label encoder that we used for training data
})
ans = model.predict(new_data)
print(ans)