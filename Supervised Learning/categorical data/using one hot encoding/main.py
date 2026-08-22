import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

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
ohe = OneHotEncoder(sparse_output=False) # it means the output will be a dense array instead of a sparse matrix
town_encoded = ohe.fit_transform(X[["town"]])

# Convert encoded array to DataFrame
town_df = pd.DataFrame(town_encoded, columns=ohe.get_feature_names_out(["town"]))

# Drop original town column
X_numeric = X.drop("town", axis=1)

# Concatenate numeric + encoded
X_final = pd.concat([X_numeric, town_df], axis=1)

print(X_final)

model = LinearRegression()

model.fit(X_final, y)
new_data = pd.DataFrame({
    "area": [1000],
    "bedrooms": [2],
    "town_Delhi": [0],
    "town_Mumbai": [1],
    "town_Pune": [0]
})

ans = model.predict(new_data)
print(ans)

